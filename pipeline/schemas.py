"""
JSON schemas for the file that stage 1 (WiLoR) and stage 2 (HaPTIC) use to talk
to each other. The two stages run in separate conda envs, so they can't share
objects in memory - stage 1 writes a JSON, stage 2 reads it back. These models
let us validate that JSON on read, so a broken file fails right at the boundary
instead of crashing somewhere deep in the pipeline.

Layout for multiple hands detected per frame (any number of hands per frame, each with its own track_id).
"""
from typing import List
from pydantic import BaseModel, Field

# --- multi-hand format with tracking (this is the one in use) ---

class HandDetection(BaseModel):
    bbox: List[float] = Field(..., min_length=4, max_length=4)  # x1, y1, x2, y2
    is_right: int = Field(..., ge=0, le=1)     # 1 = is right hand, 0 = is left hand
    conf: float = Field(..., ge=0.0, le=1.0)   # YOLO confidence score
    track_id: int = Field(..., ge=0)           # stays the same for one hand across frames


class FrameBboxMulti(BaseModel):
    img_name: str                                              # the image/frame name (ex frame_001.jpg)
    hands: List[HandDetection] = Field(default_factory=list)   # empty if no hand in the frame


class SequenceBboxesMulti(BaseModel):
    orig_w: float = Field(..., gt=0)    # image width in pixels
    orig_h: float = Field(..., gt=0)    # image height in pixels
    frames: List[FrameBboxMulti]        # list with the data collected about the detected hands for each frame
