// Shared API types, kept in sync with backend Pydantic schemas.

export interface User {
  id: string;
  email: string;
  display_name: string;
  jobs_today: number;
  jobs_quota_per_day: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export type ProjectStatus =
  | "pending"
  | "ingesting"
  | "extracting_audio"
  | "transcribing"
  | "diarizing"
  | "analyzing"
  | "segmenting"
  | "scoring"
  | "reframing"
  | "subtitling"
  | "rendering"
  | "completed"
  | "failed";

export type SourceType = "upload" | "youtube";
export type LayoutMode = "auto" | "single_focus" | "split_screen" | "fit_blur" | "fill_zoom";

export interface ClipWord {
  word: string;
  start: number;
  end: number;
  speaker?: string | null;
  confidence?: number;
}

export interface Clip {
  id: string;
  rank: number;
  start: number;
  end: number;
  duration: number;
  viral_score: number;
  category_primary: string;
  category_secondary: string;
  scores?: Record<string, number> | null;
  reasons?: string[] | null;
  title: string;
  description: string;
  hashtags: string;
  transcript: string;
  words?: ClipWord[] | null;
  subtitle_style: string;
  layout_mode: string;
  target_fps: number;
  rendered: boolean;
  render_error?: string | null;
  video_url?: string | null;
  thumb_url?: string | null;
}

export interface Project {
  id: string;
  status: ProjectStatus;
  stage_message: string;
  stage_progress: number;
  error?: string | null;
  source_type: SourceType;
  source_name: string;
  duration?: number | null;
  width?: number | null;
  height?: number | null;
  fps?: number | null;
  detected_language?: string | null;
  num_speakers: number;
  created_at: string;
  clips: Clip[];
}

export interface ProjectCreate {
  source_type: SourceType;
  source_name: string;
  source_url?: string;
  language?: string;
  target_dur_min?: number;
  target_dur_max?: number;
  target_clip_count?: number;
  desired_categories?: string[];
  subtitle_style?: string;
  target_fps?: number;
  layout_mode?: LayoutMode;
}

export type SubtitlePosition = "bottom" | "center" | "top";

/** One subtitle preset, as returned by GET /api/subtitle-styles. */
export interface SubtitleStyle {
  id: string;
  name: string;
  font: string;
  font_size: number;
  primary_color: string;
  secondary_color: string;
  outline_color: string;
  back_color: string;
  bold: boolean;
  italic: boolean;
  uppercase: boolean;
  alignment: number;
  margin_v: number;
  margin_l: number;
  margin_r: number;
  words_per_line: number;
  max_line_chars: number;
  animation: string;
  animation_intensity: number;
  position: SubtitlePosition;
  outline: number;
  shadow: number;
  keyword_color: string;
  keyword_bold: boolean;
  emoji: boolean;
  border_style: number;
  scale_x: number;
  scale_y: number;
  spacing: number;
  rotation: number;
}

export type ClipUpdate = {
  title?: string;
  start?: number;
  end?: number;
  subtitle_style?: string;
  layout_mode?: string;
  target_fps?: number;
  transcript?: string;
  words?: ClipWord[];
};

export interface PipelineEvent {
  type: "stage" | "progress" | "log" | "error" | "done" | "clip";
  data: Record<string, unknown>;
  ts: number;
}

export const PIPELINE_STAGES: { key: string; label: string }[] = [
  { key: "ingesting", label: "Ingesting video" },
  { key: "extracting_audio", label: "Extracting audio" },
  { key: "transcribing", label: "Transcribing speech" },
  { key: "diarizing", label: "Detecting speakers" },
  { key: "analyzing", label: "Detecting viral moments" },
  { key: "scoring", label: "Scoring & ranking clips" },
  { key: "reframing", label: "Reframing to 9:16" },
  { key: "subtitling", label: "Generating subtitles" },
  { key: "rendering", label: "Rendering clips" },
];
