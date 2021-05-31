"""
Common boilerplate styles/classes for Jupyter notebooks.
"""
import ipywidgets

# For slider + text combo widgets
slider_style = {"description_width": "200px"}
slider_layout = ipywidgets.Layout(width="80%")
slider_text_layout = ipywidgets.Layout(width="75px")

# For modifying the Jupyter output area
# - Limit the height of most output areas for readability
# - Prevent vertical scrollbars in nested output subareas
jupyter_output_style = """
<style>
    div.cell > div.output_wrapper > div.output.output_scroll {
        height: auto;
    }
    
    .jupyter-widgets-output-area .output_scroll {
        height: unset;
        border-radius: unset;
        -webkit-box-shadow: unset;
        box-shadow: unset;
    }
    
    .jupyter-widgets-output-area, .output_stdout, .output_result {
        height: auto;
        max-height: 65em;
        overflow-y: auto;
    }
    .jupyter-widgets-output-area .jupyter-widgets-output-area {
        max-height: unset;
    }
    
    .jupyter-widgets-view.output_subarea {
        padding: 0.4em 0;
    }
</style>
"""
