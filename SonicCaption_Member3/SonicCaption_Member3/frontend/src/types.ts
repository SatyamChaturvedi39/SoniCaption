export type TimelineItem = {id:string; start:number; end:number; category:string; label:string; display_text:string; confidence:number|null}
export type Analysis = {filename:string; duration:number; sample_rate:number; mode:string; speech_segments:any[]; sound_events:any[]; timeline:TimelineItem[]; metrics:Record<string,any>}
