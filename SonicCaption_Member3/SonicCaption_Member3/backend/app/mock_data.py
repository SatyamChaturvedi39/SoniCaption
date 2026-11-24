def analysis(filename: str, duration: float = 12.5):
    # Fixed illustrative content; timestamps are clipped to the submitted duration.
    end1 = min(4.2, max(1.0, duration * 0.34))
    event_start = min(4.3, end1 + 0.1)
    event_end = min(6.7, max(event_start + 0.2, duration * 0.55))
    speech2_start = min(6.8, event_end + 0.1)
    speech2_end = max(speech2_start + 0.2, duration - 0.3)
    return {
        "filename": filename,
        "duration": duration,
        "sample_rate": 16000,
        "mode": "mock",
        "speech_segments": [
            {"id": 1, "start": 0.5, "end": end1,
             "text": "Good morning everyone, welcome to the SonicCaption product launch.",
             "words": [
                 {"word": "Good", "start": 0.5, "end": 0.8},
                 {"word": "morning", "start": 0.9, "end": 1.4},
                 {"word": "everyone", "start": 1.5, "end": 2.1}
             ]},
            {"id": 2, "start": speech2_start, "end": speech2_end,
             "text": "Today we demonstrate how multi-modal acoustic captioning enhances accessibility.",
             "words": []}
        ],
        "sound_events": [
            {"start": event_start, "end": event_end, "label": "clapping",
             "confidence": 0.94, "model_source": "demo"}
        ],
        "timeline": [
            {"id": "t1", "start": 0.5, "end": end1, "category": "speech",
             "label": "Speech",
             "display_text": "Good morning everyone, welcome to the SonicCaption product launch.",
             "confidence": None},
            {"id": "t2", "start": event_start, "end": event_end, "category": "sound_event",
             "label": "Clapping / Applause", "display_text": "[Applause]", "confidence": 0.94},
            {"id": "t3", "start": speech2_start, "end": speech2_end, "category": "speech",
             "label": "Speech",
             "display_text": "Today we demonstrate how multi-modal acoustic captioning enhances accessibility.",
             "confidence": None}
        ],
        "metrics": {"vad_speech_ratio": 0.69, "transcription_time_ms": 520,
                    "cnn_inference_time_ms": 75, "note": "Illustrative mock metrics"}
    }

DTW = {
    "distance": 14.82, "normalized_distance": 0.185, "path_length": 7,
    "path": [[0,0],[1,1],[2,1],[3,2],[4,3],[5,4],[6,5]],
    "cost_matrix_sample": [[0.1,1.2,2.5],[1.4,0.3,1.8],[2.8,1.5,0.4]]
}
HMM = {
    "log_likelihood": -14.285,
    "best_path": ["Sunny","Sunny","Rainy","Rainy","Sunny"],
    "state_probabilities": [
        {"step":0,"state":"Sunny","prob":0.6},{"step":0,"state":"Rainy","prob":0.4},
        {"step":1,"state":"Sunny","prob":0.72},{"step":1,"state":"Rainy","prob":0.28}
    ],
    "trellis": [
        {"from":"Sunny","to":"Sunny","prob":0.7,"chosen":True},
        {"from":"Sunny","to":"Rainy","prob":0.3,"chosen":False}
    ]
}
