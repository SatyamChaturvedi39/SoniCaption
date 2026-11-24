from pydantic import BaseModel
from typing import Optional, List, Any

class Word(BaseModel):
    word: str
    start: float
    end: float

class SpeechSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: List[Word] = []

class SoundEvent(BaseModel):
    start: float
    end: float
    label: str
    confidence: Optional[float] = None
    model_source: Optional[str] = "demo"

class TimelineItem(BaseModel):
    id: str
    start: float
    end: float
    category: str
    label: str
    display_text: str
    confidence: Optional[float] = None

class AnalysisResponse(BaseModel):
    filename: str
    duration: float
    sample_rate: int
    mode: str = "mock"
    speech_segments: List[SpeechSegment]
    sound_events: List[SoundEvent]
    timeline: List[TimelineItem]
    metrics: dict[str, Any]
