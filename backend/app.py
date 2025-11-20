import streamlit as st
import os
import shutil
from agent_workflow import run_agent_workflow
from orchestrator import run_orchestrator
from profiling import check_compatibility
from schema_registry import analyze_netcdf_schema
from semantic_layer import resolve_concepts_for_schema # <--- New Import

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
                    with st.spinner("Analyzing Schema & Vectors..."):
                        # CALL THE NEW REGISTRY FUNCTION
                        # This extracts vars and auto-detects wsh_x/wsh_y pairs
                        schema = analyze_netcdf_schema(file_path)
                        
                        # Store this schema in session state
                        # Store this schema in session state
                        # st.session_state.metadata[baseline_file.name] = schema
                        
                        # 2. Level 2: Get Semantic Layer
                        # This figures out if "velocity" is possible in this specific file
                        concepts = resolve_concepts_for_schema(schema)
                        
                        # Store both
                        st.session_state.metadata[baseline_file.name] = {
                            "schema": schema,
                            "concepts": concepts,
                            "filename": baseline_file.name # Ensure filename is at top level for convenience
                        }
                        
                        st.session_state.file_paths[baseline_file.name] = file_path
                        
                        # Default analysis to baseline
                        if not st.session_state.analysis:
                            st.session_state.analysis = schema
                        
                        st.success(f"Loaded {baseline_file.name}")
                        
                        # (Optional Debug) Show the user what concepts we found
                        if schema.get("derived_concepts"):
                            st.info(f"Auto-Detected Concepts: {[c['concept_name'] for c in schema['derived_concepts']]}")

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
                    with st.spinner("Analyzing Schema & Vectors..."):
                        schema = analyze_netcdf_schema(file_path)
                        # 2. Level 2: Get Semantic Layer
                        concepts = resolve_concepts_for_schema(schema)
                        
                        st.session_state.metadata[scenario_file.name] = {
                            "schema": schema,
                            "concepts": concepts,
                            "filename": scenario_file.name
                        }
                        st.session_state.file_paths[scenario_file.name] = file_path
                        st.success(f"Loaded Scenario: {scenario_file.name}")
                        
                        if schema.get("derived_concepts"):
                            st.info(f"Auto-Detected Concepts: {[c['concept_name'] for c in schema['derived_concepts']]}")

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
            # Handle new metadata structure safely
            if isinstance(profile, dict) and "schema" in profile:
                display_profile = profile["schema"]
            else:
                display_profile = profile

            # Fallback if display_profile is None or empty
            if not display_profile:
                st.error("Could not load profile data.")
                st.stop()
            
            st.markdown(f"**File:** `{display_profile.get('filename', 'Unknown')}`")
            
            # Safe Time Horizon Access
            if "time_horizon" in display_profile:
                 # Handle the dict structure of time_horizon if present
                 th = display_profile['time_horizon']
                 if isinstance(th, dict):
                     st.markdown(f"**Time Horizon:** {th.get('start')} to {th.get('end')}")
                     st.markdown(f"**Time Steps:** {th.get('steps')}")
                 else:
                     st.markdown(f"**Time Horizon:** {th}")
            
            if "max_depth" in profile:
                st.markdown(f"**Max Depth:** {profile['max_depth']:.2f} m")
            if "elevation_range" in profile:
                st.markdown(f"**Elevation Range:** {profile['elevation_range'][0]:.2f}m to {profile['elevation_range'][1]:.2f}m")
                
            st.markdown("**Variables:**")
            vars_list = list(display_profile.get("variables", {}).keys())
            st.code(", ".join(vars_list[:10]) + ("..." if len(vars_list) > 10 else ""))

        with col2:
            if "preview_image" in profile:
                st.image(f"data:image/png;base64,{profile['preview_image']}", caption="Domain Bathymetry")
            elif "preview_error" in profile:
                st.warning(f"Could not generate preview: {profile['preview_error']}")

        # Smart Suggestions (Template based)
        st.markdown("---")
        st.markdown("**Smart Suggestions:**")
        suggestions = []
        st.markdown("**Smart Suggestions:**")
        suggestions = []
        
        # Use display_profile variables
        vars_keys = display_profile.get("variables", {}).keys()
        
        if "elev" in vars_keys:
            suggestions.append("Plot the average water surface elevation across all time steps.")
            suggestions.append("What is the maximum elevation?")
        if "hvel_x" in vars_keys and "hvel_y" in vars_keys:
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
            
            # 1. Prepare the Bundle
            metadata_bundle = {}
            
            # Get Baseline Schema
            if st.session_state.baseline_path:
                base_name = os.path.basename(st.session_state.baseline_path)
                metadata_bundle['baseline'] = st.session_state.metadata.get(base_name)
                
            # Get Scenario Schema (If it exists)
            if st.session_state.scenario_path:
                scen_name = os.path.basename(st.session_state.scenario_path)
                metadata_bundle['scenario'] = st.session_state.metadata.get(scen_name)
            
            # Run Orchestrator
            try:
                result = run_orchestrator(
                    st.session_state.messages[-1]["content"], 
                    metadata_bundle, # <--- This is the dictionary of schemas
                    st.session_state.baseline_path, 
                    st.session_state.scenario_path
                )
                
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
