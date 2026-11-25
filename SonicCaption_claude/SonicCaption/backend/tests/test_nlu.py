"""Scenario detection, per-speaker topics and speaker-assigned tasks."""
from soniccaption.services.nlu import analyse_conversation


def U(t, spk, text):
    return {"start": t, "end": t + 4, "speaker_id": spk, "text": text}


MEETING = [
    U(0, 1, "Good morning everyone. This is Priya. Let's go over the sprint status and the deadline for the release."),
    U(6, 2, "Hi, I'm Rahul. I'll make the changes to the repository by Friday and push them for review."),
    U(12, 1, "Great. Rahul, can you also update the documentation?"),
    U(16, 2, "Sure, I'll do that tomorrow."),
    U(20, 3, "I need to fix the login bug before the demo. Can you send me the test report, Priya?"),
    U(26, 1, "Yes, I'll send it today."),
    U(30, 3, "We decided to go with the new design. The backend deployment is blocked on the database migration."),
]


def test_meeting_scenario_and_speaker_names():
    r = analyse_conversation(MEETING, use_llm=False)
    assert r["scenario"]["label"] == "Meeting"
    assert r["names"] == {"1": "Priya", "2": "Rahul"}


def test_tasks_are_assigned_to_the_right_speaker():
    r = analyse_conversation(MEETING, use_llm=False)
    by_owner = {}
    for t in r["tasks"]:
        by_owner.setdefault(t["owner_id"], []).append(t)
    repo = next(t for t in by_owner[2] if "repository" in t["task"].lower())
    assert repo["due"] == "Friday" and repo["kind"] == "commitment"
    docs = next(t for t in by_owner[2] if "documentation" in t["task"].lower())      # asked by Speaker 1, accepted by Rahul
    assert docs["requested_by"].startswith("Speaker 1") and docs["due"] == "Tomorrow"
    assert any("login bug" in t["task"].lower() for t in by_owner[3])
    report = next(t for t in by_owner[1] if "report" in t["task"].lower())
    assert report["due"] == "Today"


def test_acknowledgements_do_not_become_tasks():
    r = analyse_conversation(MEETING, use_llm=False)
    assert not any(t["task"].lower().startswith(("do that", "that")) for t in r["tasks"])
    assert len(r["tasks"]) == 4


def test_negations_and_questions_create_no_tasks():
    r = analyse_conversation([U(0, 1, "I can't make the meeting tomorrow."),
                              U(4, 2, "I will not be there either. I think it's fine. I'll see how it goes. Let me know."),
                              U(8, 1, "What time is lunch?")], use_llm=False)
    assert r["tasks"] == []


def test_phone_call_support_and_talk_scenarios():
    call = [U(0, 1, "Hello? Can you hear me okay?"), U(3, 2, "Yes. Hi, this is Arun speaking."),
            U(7, 1, "Hi Arun. I'm calling about my order, I want a refund for the broken item."),
            U(12, 2, "I'm sorry to hear that. I'll process the refund today and send you a confirmation email.")]
    r = analyse_conversation(call, use_llm=False)
    assert r["scenario"]["label"] in ("Customer support call", "Phone call")
    assert any(t["owner_id"] == 2 and "refund" in t["task"].lower() for t in r["tasks"])
    talk = analyse_conversation([U(0, 1, "Welcome everyone, in this talk I'll walk you through the key takeaways. Next slide please.")], use_llm=False)
    assert talk["scenario"]["label"] == "Presentation / talk" and talk["tasks"] == []


def test_per_speaker_topics_differ():
    r = analyse_conversation(MEETING, use_llm=False)
    topics = {p["speaker_id"]: p["topics"] for p in r["speakers"]}
    assert "repository" in topics[2] and "deployment" in topics[3]
    assert r["decisions"] and r["decisions"][0]["speaker"].startswith("Speaker 3")


def test_empty_and_placeholder_input_is_reported_not_invented():
    r = analyse_conversation([{"start": 0, "end": 2, "speaker_id": 1, "text": "[Speech detected (0.00s - 2.00s)]"}], use_llm=False)
    assert r["tasks"] == [] and r["speakers"] == [] and r["warnings"]
