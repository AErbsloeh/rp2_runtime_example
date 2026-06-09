from pylsl import StreamInfo, StreamOutlet, StreamInlet,FOREVER, IRREGULAR_RATE, resolve_bypred ,cf_int8, cf_int16, cf_int32, cf_int64, cf_float32, cf_double64 ,proc_threadsafe
from numpy import issubdtype, dtype, float32, floating, int8, int16, int32, integer

class LSLHandler:
    @staticmethod
    def create_lsl_outlet(name: str, number_of_channels: int, channel_format= dtype , sampling_rate: float = IRREGULAR_RATE, type: str = "data") -> StreamOutlet:
        """Create an LSL StreamOutlet

        Args:
            name (str): Name of the LSL Stream, this is the searchable layer name
            number_of_channels (int): Number of channels in the LSL stream
            channel_format (dtype): Format of the channel data
            sampling_rate (float, optional): Sampling rate of the stream. Defaults to IRREGULAR_RATE
            type (str, optional): Type of the LSL stream. Defaults to "data"

        Returns:
            StreamOutlet: The created LSL StreamOutlet object
        """
        if issubdtype(channel_format, float32):
            lsl_format = cf_float32
        elif issubdtype(channel_format, floating):
            lsl_format = cf_double64
        elif issubdtype(channel_format, int8):
            lsl_format = cf_int8
        elif issubdtype(channel_format, int16):
            lsl_format = cf_int16
        elif issubdtype(channel_format, int32):
            lsl_format = cf_int32
        elif issubdtype(channel_format, integer):
            lsl_format = cf_int64
        else:
            # Fallback to double if the provided channel format is not recognized
            lsl_format = cf_double64
            Warning(f"Provided channel format {channel_format} is not recognized, falling back to double precision float (cf_double64) for LSL stream")

        info = StreamInfo(name=name,
                        type=type,
                        channel_count=number_of_channels, 
                        nominal_srate=sampling_rate,
                        channel_format=lsl_format,
                        source_id=name + '_uid')
        return StreamOutlet(info)
    

    @staticmethod
    def connect_to_lsl_stream(lsl_layer_name: str) -> StreamInlet:
        """Search for an LSL streams by name and connecting to them
        
        Args:    
            lsl_layer_name (str): The name of the LSL stream layer to search for

        Raises:
            RuntimeError: If no stream with the specified layer name is found

        Returns:
            StreamInlet: A connected StreamInlet object
        """        
        print("Search for LSL Stream..")
        streams = resolve_bypred(predicate=f"name='{lsl_layer_name}'")
        if streams:
            print(f"LSL Stream '{lsl_layer_name}' found, connecting...")
            inlet = StreamInlet(streams[0],
                                max_buflen= 60,
                                max_chunklen= 1024,
                                recover=True,
                                processing_flags=proc_threadsafe)
        else:
            raise RuntimeError(f"No Stream with Layer Name {lsl_layer_name} found!")
        return inlet