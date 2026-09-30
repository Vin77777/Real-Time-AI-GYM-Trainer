import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import (
    load_css,
    inject_local_font,
    inject_webrtc_styles,
)
from services.persistence.exercise_repository import (
    init_db,
    get_users_exercises,
)
from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update

from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

STATIC_DIR = APP_DIR / "static"
ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# ENVIRONMENT
# ============================================================

# Load .env from:
# Real-Time AI Gym Trainer/.env
load_dotenv(ENV_FILE)


# ============================================================
# HELPERS
# ============================================================

def initialize_voice_pipeline():
    """
    Create the Groq + LLM + TTS voice pipeline.

    Returns:
        VoicePipeline object if successful.
        None if initialization fails.
    """

    try:
        # Reload environment variables in case .env was changed
        load_dotenv(ENV_FILE, override=False)

        api_key = os.getenv("GROQ_API_KEY", "").strip()

        # Streamlit Cloud secrets fallback
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "").strip()
            except Exception:
                api_key = ""

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Check your .env file or Streamlit secrets."
            )

        # Create Groq client
        groq_client = Groq(api_key=api_key)

        # Create AI coach
        llm_coach = LLMCoach(groq_client)

        # Create text-to-speech
        tts = TextToSpeech()

        # Create complete voice pipeline
        pipeline = VoicePipeline(
            llm=llm_coach,
            tts=tts,
        )

        return pipeline

    except Exception as e:
        st.session_state.voice_error = str(e)
        return None


def trigger_coach_event(event, exercise, metrics=None):
    """
    Safely send an event to the AI coach.

    Prevents the entire Streamlit application from crashing
    if Groq or TTS fails.
    """

    if metrics is None:
        metrics = {}

    voice_pipeline = st.session_state.get("voice_pipeline")

    if voice_pipeline is None:
        return

    try:
        result = voice_pipeline.process_event(
            event=event,
            exercise=exercise,
            metrics=metrics,
        )

        if result:
            audio_bytes, feedback_text = result

            if audio_bytes:
                st.session_state.audio_to_play = audio_bytes

            if feedback_text:
                st.session_state.coach_feedback = feedback_text

    except Exception as e:
        st.session_state.voice_runtime_error = str(e)


# ============================================================
# MAIN APP
# ============================================================

def main():

    # --------------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------------

    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="AI Real-time GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered",
    )

    # --------------------------------------------------------
    # LOAD CSS / FONTS
    # --------------------------------------------------------

    load_css(str(STATIC_DIR / "style.css"))

    inject_local_font(
        str(STATIC_DIR / "AdobeClean.otf"),
        "AdobeClean",
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    init_db()

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if not render_login_wall():
        return

    # --------------------------------------------------------
    # SESSION DEFAULTS
    # --------------------------------------------------------

    initial_session_defaults()

    # Additional voice-related session values
    if "voice_pipeline" not in st.session_state:
        st.session_state.voice_pipeline = initialize_voice_pipeline()

    if "audio_to_play" not in st.session_state:
        st.session_state.audio_to_play = None

    if "coach_feedback" not in st.session_state:
        st.session_state.coach_feedback = None

    if "voice_error" not in st.session_state:
        st.session_state.voice_error = None

    if "voice_runtime_error" not in st.session_state:
        st.session_state.voice_runtime_error = None

    # --------------------------------------------------------
    # WORKOUT STATUS
    # --------------------------------------------------------

    workout_started = st.session_state.get(
        "workout_started",
        False,
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title("🏋️‍♂️ Apna AI Coach")

        username = st.session_state.get("username")

        if username:
            st.caption(f"👤 Login as {username}")

        st.divider()

        st.subheader("Workout Plan")

        # ----------------------------------------------------
        # BEFORE WORKOUT
        # ----------------------------------------------------

        if not workout_started:

            plan_exercise = st.selectbox(
                "Exercise",
                options=EXERCISE_OPTIONS,
                key="plan_exercise",
            )

            plan_sets = st.number_input(
                "Sets",
                min_value=1,
                max_value=50,
                step=1,
                key="plan_sets",
            )

            plan_reps = st.number_input(
                "Reps per Set",
                min_value=1,
                max_value=50,
                step=1,
                key="plan_reps",
            )

            start_session_button = st.button(
                "Start Workout",
                width="stretch",
                key="start_session_button",
            )

            if start_session_button:

                # --------------------------------------------
                # SET WORKOUT STATE
                # --------------------------------------------

                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)

                st.session_state.reps = 0
                st.session_state.sets_completed = 0
                st.session_state.current_set_reps = 0

                st.session_state.workout_complete = False
                st.session_state.last_saved_sets_completed = 0
                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False

                st.session_state.set_cycle_started_at = time.time()

                st.session_state.workout_started = True

                # Clear old feedback/audio
                st.session_state.audio_to_play = None
                st.session_state.coach_feedback = None
                st.session_state.voice_runtime_error = None

                # --------------------------------------------
                # AI START MESSAGE
                # --------------------------------------------

                trigger_coach_event(
                    event="workout_started",
                    exercise=plan_exercise,
                    metrics={},
                )

                # Re-run app
                st.rerun()

        # ----------------------------------------------------
        # DURING WORKOUT
        # ----------------------------------------------------

        else:

            exercise = st.session_state.get(
                "exercise_type",
                "Squats",
            )

            sets = st.session_state.get(
                "target_sets",
                3,
            )

            reps = st.session_state.get(
                "reps_per_set",
                10,
            )

            st.info(
                f"**{exercise}** — {sets} Sets / {reps} Reps"
            )

            end_session_button = st.button(
                "End Workout",
                key="end_session_button",
                width="stretch",
            )

            if end_session_button:

                # --------------------------------------------
                # AI WORKOUT COMPLETION MESSAGE
                # --------------------------------------------

                trigger_coach_event(
                    event="workout_completed",
                    exercise=exercise,
                    metrics={},
                )

                st.session_state.workout_started = False

                st.rerun()

        # ====================================================
        # PROGRESS
        # ====================================================

        if workout_started:

            st.divider()

            exercise = st.session_state.get(
                "exercise_type",
                "Squats",
            )

            total_reps = st.session_state.get(
                "reps",
                0,
            )

            current_set_reps = st.session_state.get(
                "current_set_reps",
                0,
            )

            reps_per_set = st.session_state.get(
                "reps_per_set",
                0,
            )

            sets_completed = st.session_state.get(
                "sets_completed",
                0,
            )

            target_sets = st.session_state.get(
                "target_sets",
                0,
            )

            st.subheader("Progress")

            st.metric(
                "Total Reps",
                f"{total_reps}",
            )

            st.metric(
                "Current Set Reps",
                f"{current_set_reps} / {reps_per_set}",
            )

            st.metric(
                "Sets Completed",
                f"{sets_completed} / {target_sets}",
            )

            # =================================================
            # EXERCISE METRICS
            # =================================================

            st.divider()

            if exercise == "Squats":

                st.subheader("Squat Metrics")

                st.metric(
                    "Knee Angle",
                    f"{st.session_state.get('knee_angle', 0)}°",
                )

                st.metric(
                    "Back Angle",
                    f"{st.session_state.get('back_angle', 0)}°",
                )

                st.metric(
                    "Depth Status",
                    st.session_state.get(
                        "depth_status",
                        "N/A",
                    ),
                )

            elif exercise == "Push-ups":

                st.subheader("Push-up Metrics")

                st.metric(
                    "Elbow Angle",
                    f"{st.session_state.get('elbow_angle', 0)}°",
                )

                st.metric(
                    "Body Alignment",
                    st.session_state.get(
                        "body_alignment",
                        "N/A",
                    ),
                )

                st.metric(
                    "Hip Position",
                    st.session_state.get(
                        "hip_status",
                        "N/A",
                    ),
                )

            elif exercise == "Biceps Curls (Dumbbell)":

                st.subheader("Curl Metrics")

                st.metric(
                    "Elbow Angle",
                    f"{st.session_state.get('elbow_angle', 0)}°",
                )

                st.metric(
                    "Shoulder Stability",
                    st.session_state.get(
                        "shoulder_status",
                        "N/A",
                    ),
                )

                st.metric(
                    "Swing Detection",
                    st.session_state.get(
                        "swing_status",
                        "N/A",
                    ),
                )

            elif exercise == "Shoulder Press":

                st.subheader("Shoulder Press Metrics")

                st.metric(
                    "Elbow Angle",
                    f"{st.session_state.get('elbow_angle', 0)}°",
                )

                st.metric(
                    "Arm Extension",
                    st.session_state.get(
                        "extension_status",
                        "N/A",
                    ),
                )

                st.metric(
                    "Back Arch",
                    st.session_state.get(
                        "back_arch_status",
                        "N/A",
                    ),
                )

            elif exercise == "Lunges":

                st.subheader("Lunge Metrics")

                st.metric(
                    "Front Knee Angle",
                    f"{st.session_state.get('front_knee_angle', 0)}°",
                )

                st.metric(
                    "Torso Angle",
                    f"{st.session_state.get('torso_angle', 0)}°",
                )

                st.metric(
                    "Balance Status",
                    st.session_state.get(
                        "balance_status",
                        "N/A",
                    ),
                )

    # ========================================================
    # MAIN CONTENT
    # ========================================================

    st.title("AI Real-time GYM Coach")

    st.markdown(
        "#### Real-time pose detection with proactive AI voice coaching"
    )

    # --------------------------------------------------------
    # VOICE STATUS
    # --------------------------------------------------------

    if st.session_state.get("voice_pipeline") is not None:

        st.success("🤖 AI Voice Coach: Ready")

    else:

        error = st.session_state.get("voice_error")

        st.warning(
            "⚠️ AI Voice Coach is not available."
        )

        if error:
            st.code(
                error,
                language="text",
            )

    # --------------------------------------------------------
    # RUNTIME VOICE ERROR
    # --------------------------------------------------------

    runtime_error = st.session_state.get(
        "voice_runtime_error"
    )

    if runtime_error:

        st.error(
            f"Voice/AI error: {runtime_error}"
        )

    # --------------------------------------------------------
    # PLAY GENERATED AUDIO
    # --------------------------------------------------------

    audio_bytes = st.session_state.get(
        "audio_to_play"
    )

    if audio_bytes:

        try:

            autoplay_audio(audio_bytes)

        except Exception as e:

            st.error(
                f"Audio playback error: {e}"
            )

    # --------------------------------------------------------
    # SHOW AI TEXT FEEDBACK
    # --------------------------------------------------------

    coach_feedback = st.session_state.get(
        "coach_feedback"
    )

    if coach_feedback:

        st.markdown("")

        st.success(
            f"🤖 **Coach:** {coach_feedback}"
        )

    # ========================================================
    # BEFORE WORKOUT
    # ========================================================

    if not workout_started:

        st.markdown(
            """
            <div style="
                border: 10px dashed #444;
                border-radius: 0px;
                padding: 48px 32px;
                text-align: center;
                color: #888;
                margin-top: 32px;
                margin-bottom: 32px;
            ">
                <h2 style="color:#ccc; margin-bottom:8px;">
                    👈 Set your workout plan
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # DURING WORKOUT
    # ========================================================

    else:

        context = webrtc_streamer(
            key="exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,

            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            },

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

        # ----------------------------------------------------
        # SYNC CAMERA METRICS
        # ----------------------------------------------------

        sync_metrics_update(context)

        # ----------------------------------------------------
        # KEEP STREAMLIT UPDATING
        # ----------------------------------------------------
        if context and context.state.playing:
             time.sleep(0.25)

        

        # ----------------------------------------------------
        # WEBRTC CSS
        # ----------------------------------------------------

        inject_webrtc_styles()

    # ========================================================
    # WORKOUT HISTORY
    # ========================================================

    st.divider()

    st.markdown("#### Workout History")

    user_id = st.session_state.get(
        "user_id",
        0,
    )

    if isinstance(user_id, int):

        history_rows = get_users_exercises(
            user_id
        )

        arr = [
            {
                "Exercise": row["exercise_name"],
                "Reps": row["reps"],
                "Sets": row["sets"],
                "Time (sec)": row["time"],
                "Date": row["created_at"],
            }
            for row in history_rows
        ]

        df = pd.DataFrame(arr)

        if not df.empty:

            df["Date"] = pd.to_datetime(
                df["Date"]
            ).dt.date

            agg_df = (
                df.groupby(
                    ["Exercise", "Date"]
                )
                .agg(
                    {
                        "Reps": "sum",
                        "Sets": "sum",
                        "Time (sec)": "sum",
                    }
                )
                .reset_index()
            )

            agg_df.index += 1

            st.table(
                agg_df,
                border="horizontal",
            )

        else:

            st.info(
                "No workout history found."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()