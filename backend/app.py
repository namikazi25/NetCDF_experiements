import streamlit as st
import os
import shutil
from agent_workflow import run_agent_workflow
from orchestrator import run_orchestrator
from profiling import generate_profile

st.set_page_config(page_title="NetCDF LLM Analyst", layout="wide")

st.title("🌍 NetCDF LLM Analyst")
st.markdown("Upload NetCDF files and ask questions about them in natural language.")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "file_paths" not in st.session_state:
    st.session_state.file_paths = {}

from profiling import generate_profile

# ... (existing imports)

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    
    # Dual Uploader for Comparison Mode
    st.subheader("1. Baseline Model")
    baseline_file = st.file_uploader("Upload Baseline .nc", type=["nc"], key="base_uploader")
    
    st.subheader("2. Scenario Model (Optional)")
    scenario_file = st.file_uploader("Upload Scenario .nc", type=["nc"], key="scen_uploader")
    
    # Process Baseline
    if baseline_file:
        if baseline_file.name not in st.session_state.metadata:
            with st.spinner(f"Processing Baseline: {baseline_file.name}..."):
                file_path = os.path.abspath(os.path.join(UPLOAD_DIR, baseline_file.name))
                with open(file_path, "wb") as f:
                    f.write(baseline_file.getvalue())
                
                try:
                    profile = generate_profile(file_path)
                    st.session_state.metadata[baseline_file.name] = profile
                    st.session_state.file_paths[baseline_file.name] = file_path
                    # Default analysis to baseline
                    if not st.session_state.analysis:
                        st.session_state.analysis = profile
                    st.success(f"Loaded Baseline: {baseline_file.name}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Process Scenario
    if scenario_file:
        if scenario_file.name not in st.session_state.metadata:
            with st.spinner(f"Processing Scenario: {scenario_file.name}..."):
                file_path = os.path.abspath(os.path.join(UPLOAD_DIR, scenario_file.name))
                with open(file_path, "wb") as f:
                    f.write(scenario_file.getvalue())
                
                try:
                    profile = generate_profile(file_path)
                    st.session_state.metadata[scenario_file.name] = profile
                    st.session_state.file_paths[scenario_file.name] = file_path
                    st.success(f"Loaded Scenario: {scenario_file.name}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Determine Mode
    if baseline_file and scenario_file:
        st.session_state.mode = "comparison"
        st.session_state.baseline_path = st.session_state.file_paths.get(baseline_file.name)
        st.session_state.scenario_path = st.session_state.file_paths.get(scenario_file.name)
        st.info("⚔️ Comparison Mode Active")
        
        # Run Compatibility Check
        from profiling import check_compatibility
        is_compat, msg = check_compatibility(st.session_state.baseline_path, st.session_state.scenario_path)
        if not is_compat:
            st.error(f"⚠️ Incompatible Models: {msg}")
            st.session_state.mode = "single" # Revert to single to prevent crashes
        elif "Warning" in msg:
            st.warning(msg)
            
    elif baseline_file:
        st.session_state.mode = "single"
        st.session_state.baseline_path = st.session_state.file_paths.get(baseline_file.name)
        st.session_state.scenario_path = None
    else:
        st.session_state.mode = "waiting"
        
    if st.session_state.metadata:
        st.subheader("Loaded Files")
        for filename in st.session_state.metadata.keys():
            st.text(f"📄 {filename}")

# Main chat interface
st.subheader("Chat Analysis")

# Display Profile Card (First Look)
if st.session_state.analysis:
    profile = st.session_state.analysis
    with st.expander("📊 Dataset Profile (First Look)", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**File:** `{profile.get('filename')}`")
            if "time_start" in profile:
                st.markdown(f"**Time Horizon:** {profile['time_start']} to {profile['time_end']}")
                st.markdown(f"**Time Steps:** {profile.get('time_steps')}")
            
            if "max_depth" in profile:
                st.markdown(f"**Max Depth:** {profile['max_depth']:.2f} m")
            if "elevation_range" in profile:
                st.markdown(f"**Elevation Range:** {profile['elevation_range'][0]:.2f}m to {profile['elevation_range'][1]:.2f}m")
                
            st.markdown("**Variables:**")
            st.code(", ".join(profile.get("variables", [])[:10]) + ("..." if len(profile.get("variables", [])) > 10 else ""))

        with col2:
            if "preview_image" in profile:
                st.image(f"data:image/png;base64,{profile['preview_image']}", caption="Domain Bathymetry")
            elif "preview_error" in profile:
                st.warning(f"Could not generate preview: {profile['preview_error']}")

        # Smart Suggestions (Template based)
        st.markdown("---")
        st.markdown("**Smart Suggestions:**")
        suggestions = []
        if "elev" in profile.get("variables", []):
            suggestions.append("Plot the average water surface elevation across all time steps.")
            suggestions.append("What is the maximum elevation?")
        if "hvel_x" in profile.get("variables", []) and "hvel_y" in profile.get("variables", []):
            suggestions.append("Calculate the maximum horizontal velocity magnitude.")
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f"sugg_{i}"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display images if present
        if "images" in message:
            for img_str in message["images"]:
                st.image(f"data:image/png;base64,{img_str}")

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

from orchestrator import run_orchestrator

# ... (existing imports)

# ... (existing setup code)

# Check if the last message is from the user and needs a response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Generate response
    if not st.session_state.metadata:
        response = "Please upload a NetCDF file first so I can analyze it."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        with st.chat_message("assistant"):
            # Container for live updates
            status_container = st.status("🤖 Multi-Agent Workflow Running...", expanded=True)
            
            # Combine metadata from all files
            combined_metadata = {"files": st.session_state.metadata}
            
            # Get paths from session state
            baseline_path = st.session_state.get("baseline_path")
            scenario_path = st.session_state.get("scenario_path")
            
            # Fallback for single file mode if paths not set (legacy support)
            if not baseline_path and st.session_state.file_paths:
                first_filename = list(st.session_state.file_paths.keys())[0]
                baseline_path = os.path.abspath(st.session_state.file_paths[first_filename])
            
            # Run Orchestrator
            try:
                result = run_orchestrator(st.session_state.messages[-1]["content"], combined_metadata, baseline_path, scenario_path)
                
                # Visualize Steps
                for step in result.get("steps_log", []):
                    status_container.write(f"**{step['stage']}**: {step['status']}")
                    if step.get("output"):
                        with status_container.expander(f"Details: {step['stage']}"):
                            st.json(step["output"])
                
                status_container.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                
                st.markdown(result["response"])
                for img_str in result["images"]:
                    st.image(f"data:image/png;base64,{img_str}")
                    
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"],
                    "images": result["images"],
                    "steps_log": result["steps_log"] # Store logs if we want to show them later
                })
                
            except Exception as e:
                status_container.update(label="❌ Workflow Failed", state="error")
                st.error(f"An error occurred: {e}")
