import json
from pathlib import Path

from ._live_visualizer import LivePlotterChannelConfig, PlotSessionConfig


def load_plot_config(config_path: Path) -> list[LivePlotterChannelConfig]:
    """
    Load the live plot configuration from a JSON file.

    Only channels whose "enabled" value is true are converted into
    LivePlotterChannelConfig objects.

    Args:
        config_path:
            Path to the JSON configuration file.

    Returns:
        A list of LivePlotterChannelConfig objects for enabled channels.

    Raises:
        FileNotFoundError:
            If the JSON configuration file does not exist.

        ValueError:
            If the configuration does not contain a valid "channels" list.
    """

    with config_path.open("r", encoding="utf-8") as config_file:
        config_data = json.load(config_file) #converting the JSON file into a Python dictionary

    channels = config_data.get("channels") # Extract the channels list from the json file

    if not isinstance(channels, list): #check if the channels is a list, if not raise an error
        raise ValueError(
            'The plot configuration must contain a "channels" list.'
        )
    display_mode = config_data.get("display_mode", "filtered") #retrieve the display mode from the config data, defaulting to "filtered" if not specified
    grid_layout = config_data.get("grid_layout", None) #retrieve the grid layout from the config data, defaulting to None if not specified

    # Create a mapping of channel numbers to their positions in the grid layout
    position_by_channel_number = {
        channel_number: (row_idx, col_idx)
        for row_idx, row in enumerate(grid_layout )
        for col_idx, channel_number in enumerate(row)
        if channel_number# only include channel numbers that are not zero
    }
    plot_configs: list[LivePlotterChannelConfig] = [] #creates an empty list to store the LivePlotterChannelConfig objects
    for channel in channels:
        channel_number = channel["channel_number"] 
        grid_row, grid_col = position_by_channel_number.get(channel_number, (None, None)) #get the row and column position of the channel in the grid layout, defaulting to None if not found

        plot_config = LivePlotterChannelConfig( #creates a LivePlotterChannelConfig object for the channel and adds it to the list of plot_configs
            name=channel["name"],
            visualized_channel=channel_number - 1,
            lsl_layer_name=channel["lsl_layer_name"],
            window_width_sec=channel["window_width_sec"],
            curve_color=channel["curve_color"],
            enabled=channel.get("enabled", True),
            grid_row=grid_row,
            grid_col=grid_col,
        )

        plot_configs.append(plot_config)

    return PlotSessionConfig(channels=plot_configs, display_mode=display_mode)