
"""
src/app.py

F1 Vision Tracker - Interactive Portfolio Dashboard
----------------------------------------------------

Purpose:
    Interactive Streamlit dashboard for exploring the outputs generated
    by the F1 computer-vision pipeline.

    The dashboard is intentionally presentation-focused:
    it communicates what the system currently implements, shows real
    detection/tracking evidence, and clearly separates experimental
    components from future development.

Run:
    streamlit run src/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"
ANNOTATED_DIR = PROJECT_ROOT / "outputs" / "detections" / "annotated"
ANNOTATED_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "annotated_videos"


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Vision Tracker",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.title("🏎️ F1 Vision Tracker")

st.markdown(
    """
### Computer Vision for Formula 1 Race Footage

An experimental computer-vision pipeline for analyzing race footage
through **object detection, multi-object tracking, and visual analytics**.

The project uses real Formula 1 broadcast footage as a challenging
computer-vision environment containing small objects, camera motion,
occlusion, packed vehicle groups, and changing viewpoints.
"""
)

st.divider()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Project")

    st.markdown(
        """
        **Core technologies**

        - Python
        - OpenCV
        - YOLOv8
        - ByteTrack
        - Pandas
        - Streamlit

        **Hardware**

        CPU-based inference
        """
    )

    st.divider()

    st.caption(
        "This dashboard visualizes generated outputs. "
        "Detection and tracking are executed by the project scripts."
    )


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

(
    tab_overview,
    tab_detection,
    tab_compare,
    tab_tracking,
    tab_identity,
    tab_video,
) = st.tabs(
    [
        "Overview",
        "Detection Results",
        "Run Comparison",
        "Tracking",
        "Identity Experiments",
        "Video Data",
    ]
)


# ===========================================================================
# OVERVIEW
# ===========================================================================

with tab_overview:

    st.header("Project Overview")

    st.markdown(
        """
        **F1 Vision Tracker** explores how modern computer-vision techniques
        can be applied to Formula 1 race footage.

        The current system establishes a working baseline for detecting
        vehicles, associating detections across consecutive frames, and
        inspecting the resulting computer-vision data.
        """
    )

    st.subheader("Pipeline")

    st.code(
        """
Race Video
     │
     ▼
Frame Extraction
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
Vehicle Candidate Analysis
     │
     ▼
ByteTrack Multi-Object Tracking
     │
     ▼
Track & Detection Analytics
     │
     ▼
Identity Experiments
     │
     ▼
Interactive Streamlit Dashboard
        """,
        language="text",
    )

    st.divider()

    # -----------------------------------------------------------------------
    # IMPLEMENTED
    # -----------------------------------------------------------------------

    st.header("What the system currently implements")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎯 Object Detection")
        st.write(
            """
            YOLOv8 is used to detect vehicle-like objects in race footage.
            Each detection records its class, confidence score, bounding box,
            and center coordinates.
            """
        )

    with col2:
        st.subheader("🔗 Multi-Object Tracking")
        st.write(
            """
            ByteTrack associates detections across consecutive frames and
            assigns persistent track IDs to observed objects.
            """
        )

    with col3:
        st.subheader("📊 Visual Analytics")
        st.write(
            """
            Detection and tracking outputs are stored as structured data
            and explored through this interactive dashboard.
            """
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("🎥 Annotated Video")
        st.write(
            """
            The pipeline generates annotated race footage showing
            detection boxes and tracking IDs for visual inspection.
            """
        )

    with col5:
        st.subheader("🧪 Experiments")
        st.write(
            """
            Different confidence, IoU, model, and detection settings can
            be compared using separate output runs.
            """
        )

    with col6:
        st.subheader("🗂️ Structured Outputs")
        st.write(
            """
            Detection and tracking information is exported to CSV files,
            providing data for downstream analysis.
            """
        )

    st.divider()

    # -----------------------------------------------------------------------
    # WHAT WAS LEARNED
    # -----------------------------------------------------------------------

    st.header("Computer-Vision Challenges Observed")

    st.markdown(
        """
        Formula 1 footage presents several challenges for a generic
        pretrained detector:

        - Cars can occupy only a small number of pixels when distant.
        - F1 cars have unusual shapes compared with conventional road
          vehicles.
        - Multiple cars can overlap or appear extremely close together.
        - Broadcast cameras frequently change viewpoint.
        - Temporary occlusion can cause tracking instability.
        - A generic COCO-trained detector may classify an F1 car as
          either **car** or **truck**.
        - Low-resolution footage limits reliable visual identification
          of individual drivers.
        """
    )

    st.info(
        "These observations are treated as part of the project's baseline "
        "evaluation rather than hidden from the results."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # PROJECT STATUS
    # -----------------------------------------------------------------------

    st.header("Project Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Implemented")

        st.markdown(
            """
            - [x] Video inspection
            - [x] Frame extraction
            - [x] YOLOv8 detection
            - [x] Confidence scoring
            - [x] Bounding-box extraction
            - [x] Car/truck class analysis
            - [x] ByteTrack tracking
            - [x] Persistent track IDs
            - [x] Tracking statistics
            - [x] CSV data generation
            - [x] Annotated video generation
            - [x] Streamlit results dashboard
            """
        )

    with col2:
        st.subheader("Experimental / Future")

        st.markdown(
            """
            - [ ] F1-specific detector fine-tuning
            - [ ] Race-surface region-of-interest filtering
            - [ ] Robust driver/car identification
            - [ ] Improved identity preservation after occlusion
            - [ ] Camera-aware track association
            - [ ] Lap detection
            - [ ] Race-position reconstruction
            - [ ] Real-world speed estimation
            """
        )

    st.divider()

    st.caption(
        "Current version: experimental baseline | "
        "Designed for evaluation, iteration, and future F1-specific modeling."
    )


# ===========================================================================
# DETECTION RESULTS
# ===========================================================================

with tab_detection:

    st.header("Detection Results")

    st.markdown(
        """
        Explore the actual outputs produced by the YOLOv8 detection stage.
        Each frame contains the model's predicted bounding boxes and
        confidence scores.
        """
    )

    if not ANNOTATED_DIR.exists():

        st.info(
            "No detection outputs found. Run detector.py to generate results."
        )

    else:

        run_folders = sorted(
            [p for p in ANNOTATED_DIR.iterdir() if p.is_dir()]
        )

        if not run_folders:

            st.info(
                "No annotated detection runs found. Run detector.py first."
            )

        else:

            run_names = [p.name for p in run_folders]

            selected_run = st.selectbox(
                "Detection run",
                run_names,
                key="detection_run",
            )

            run_path = ANNOTATED_DIR / selected_run

            images = sorted(run_path.glob("*.jpg"))

            if not images:

                st.warning("This run does not contain annotated JPG frames.")

            else:

                st.write(f"**Frames available:** {len(images)}")

                idx = st.slider(
                    "Frame",
                    0,
                    len(images) - 1,
                    0,
                    key="detection_slider",
                )

                st.image(
                    str(images[idx]),
                    caption=images[idx].name,
                    width="stretch",
                )

                st.caption(
                    f"Frame {idx + 1} of {len(images)}"
                )


# ===========================================================================
# RUN COMPARISON
# ===========================================================================

with tab_compare:

    st.header("Run Comparison")

    st.markdown(
        """
        Compare two detection runs using the same frame index.

        This is useful when evaluating changes to parameters such as
        confidence threshold, IoU, model size, or image scaling.
        """
    )

    if not ANNOTATED_DIR.exists():

        st.info("No detection runs available.")

    else:

        run_folders = sorted(
            [p for p in ANNOTATED_DIR.iterdir() if p.is_dir()]
        )

        if len(run_folders) < 1:

            st.info("No detection runs available.")

        else:

            run_names = [p.name for p in run_folders]

            col_a, col_b = st.columns(2)

            with col_a:

                run_a = st.selectbox(
                    "Run A",
                    run_names,
                    index=0,
                    key="comparison_run_a",
                )

            with col_b:

                run_b = st.selectbox(
                    "Run B",
                    run_names,
                    index=min(1, len(run_names) - 1),
                    key="comparison_run_b",
                )

            images_a = sorted(
                (ANNOTATED_DIR / run_a).glob("*.jpg")
            )

            images_b = sorted(
                (ANNOTATED_DIR / run_b).glob("*.jpg")
            )

            max_idx = max(
                len(images_a),
                len(images_b),
            ) - 1

            if max_idx >= 0:

                idx = st.slider(
                    "Frame index",
                    0,
                    max_idx,
                    0,
                    key="comparison_slider",
                )

                col_a, col_b = st.columns(2)

                with col_a:

                    st.subheader(run_a)

                    if idx < len(images_a):

                        st.image(
                            str(images_a[idx]),
                            caption=images_a[idx].name,
                            width="stretch",
                        )

                    else:

                        st.warning("No frame at this index.")

                with col_b:

                    st.subheader(run_b)

                    if idx < len(images_b):

                        st.image(
                            str(images_b[idx]),
                            caption=images_b[idx].name,
                            width="stretch",
                        )

                    else:

                        st.warning("No frame at this index.")


# ===========================================================================
# TRACKING
# ===========================================================================

with tab_tracking:

    st.header("Multi-Object Tracking")

    st.markdown(
        """
        ByteTrack is used to associate detections between consecutive
        frames. Each observed object receives a `track_id`, allowing its
        movement through the video to be analyzed over time.
        """
    )

    track_csvs = (
        sorted(TRACKING_DIR.glob("tracks_*.csv"))
        if TRACKING_DIR.exists()
        else []
    )

    if not track_csvs:

        st.info(
            "No tracking CSVs found. Run tracker.py to generate tracking data."
        )

    else:

        csv_names = [p.name for p in track_csvs]

        selected_csv = st.selectbox(
            "Tracking dataset",
            csv_names,
            key="tracking_csv",
        )

        df = pd.read_csv(TRACKING_DIR / selected_csv)

        if df.empty:

            st.warning("The selected tracking file contains no data.")

        else:

            track_lengths = (
                df.groupby("track_id")
                .size()
                .sort_values(ascending=False)
            )

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Unique Track IDs",
                int(df["track_id"].nunique()),
            )

            col2.metric(
                "Total Detections",
                int(len(df)),
            )

            col3.metric(
                "Longest Track",
                f"{int(track_lengths.max())} frames",
            )

            col4.metric(
                "Short Tracks",
                int((track_lengths <= 3).sum()),
            )

            st.divider()

            st.subheader("Track Persistence")

            st.bar_chart(track_lengths)

            st.caption(
                "Short-lived tracks can indicate fragmented tracking or "
                "transient detections and should be interpreted alongside "
                "the annotated video."
            )

            st.divider()

            st.subheader("Tracking Data")

            st.dataframe(
                df,
                width="stretch",
                height=450,
            )


# ===========================================================================
# IDENTITY EXPERIMENTS
# ===========================================================================

with tab_identity:

    st.header("Identity Experiments")

    st.markdown(
        """
        This section contains experiments exploring how computer-vision
        tracks could eventually be associated with individual race cars
        or drivers.

        These experiments are **not presented as reliable driver
        identification** in the current version.
        """
    )

    st.info(
        "The current race footage resolution limits reliable identification "
        "of individual drivers from visual details alone."
    )

    # -----------------------------------------------------------------------
    # COLOR SIGNATURES
    # -----------------------------------------------------------------------

    color_csvs = (
        sorted(ANALYTICS_DIR.glob("track_colors_*.csv"))
        if ANALYTICS_DIR.exists()
        else []
    )

    if color_csvs:

        st.subheader("Vehicle Appearance / Color Signatures")

        selected = st.selectbox(
            "Color signature dataset",
            [p.name for p in color_csvs],
            key="identity_color_csv",
        )

        df = pd.read_csv(ANALYTICS_DIR / selected)

        for _, row in df.iterrows():

            col_swatch, col_info = st.columns([1, 5])

            with col_swatch:

                if "dominant_color_hex" in row:

                    st.markdown(
                        f"""
                        <div style="
                            width:50px;
                            height:50px;
                            background:{row["dominant_color_hex"]};
                            border-radius:8px;
                            border:1px solid #888;
                        "></div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_info:

                color = row.get(
                    "dominant_color_hex",
                    "N/A",
                )

                samples = row.get(
                    "sample_count",
                    "N/A",
                )

                st.write(
                    f"**Track {row['track_id']}** — "
                    f"{color} — {samples} samples"
                )

    else:

        st.subheader("Vehicle Appearance / Color Signatures")

        st.caption(
            "No color-signature experiment results are available yet."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # GRID MAPPING
    # -----------------------------------------------------------------------

    mapping_path = ANALYTICS_DIR / "grid_mapping.csv"

    st.subheader("Starting-Grid Identity Experiment")

    if mapping_path.exists():

        mdf = pd.read_csv(mapping_path)

        st.dataframe(
            mdf,
            width="stretch",
        )

        st.caption(
            "Experimental mapping between tracked objects and known "
            "starting-grid information."
        )

    else:

        st.caption(
            "No grid-mapping results are available yet."
        )


# ===========================================================================
# VIDEO DATA
# ===========================================================================

with tab_video:

    st.header("Video Data")

    st.markdown(
        """
        Metadata collected during video inspection and preprocessing.
        This information is used to understand frame rate, resolution,
        duration, and the structure of the input footage.
        """
    )

    info_path = REPORTS_DIR / "video_info.json"

    if not info_path.exists():

        st.info(
            "No video metadata found. Run video_info.py first."
        )

    else:

        try:

            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rows = [
                v
                for v in data.get("videos", [])
                if v.get("opened_successfully")
            ]

            if rows:

                columns = [
                    "file",
                    "fps",
                    "total_frames",
                    "width",
                    "height",
                    "duration_hms",
                ]

                available_columns = [
                    c for c in columns if c in rows[0]
                ]

                df = pd.DataFrame(rows)[available_columns]

                if "file" in df.columns:

                    df["file"] = df["file"].apply(
                        lambda p: Path(p).name
                    )

                st.dataframe(
                    df,
                    width="stretch",
                )

            else:

                st.info(
                    "video_info.json contains no successfully opened videos."
                )

        except Exception as exc:

            st.error(
                f"Could not read video_info.json: {exc}"
            )


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "F1 Vision Tracker • Experimental Computer Vision Project • "
    "YOLOv8 + ByteTrack + OpenCV + Streamlit"
)
