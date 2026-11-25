module TEST_ENVIRONMENT#(
    parameter integer BITWIDTH_DATA = 16,
    parameter integer BITWIDTH_ADR = 6,
    parameter integer NUM_DUT = 1,
    parameter integer NUM_BITS_HEADER = 32,
    parameter integer UINT_DATATYPE = 1
)(
    input wire CLK,
    input wire RSTN,
    input wire START_FLAG,
    input wire [$clog2(NUM_DUT):0] SEL,
    input wire [BITWIDTH_ADR-'d1:0] ADR,
    input wire RnW,
    input wire [BITWIDTH_DATA-'d1:0] DATA_IN,
    output wire [BITWIDTH_DATA-'d1:0] DATA_OUT,
    output wire [NUM_BITS_HEADER-'d1:0] HEAD_INFO, 
    output wire RDY_FLAG
);

    localparam BITWIDTH_HEAD = NUM_BITS_HEADER - 'd6;
    wire [BITWIDTH_DATA-'d1:0] dout [NUM_DUT:0];
    assign DATA_OUT = dout[SEL]; 

    wire [NUM_DUT:0] rdy_filt;
    assign RDY_FLAG = rdy_filt[SEL];

    wire [NUM_BITS_HEADER-'d7:0] head_array [NUM_DUT:0];
    assign HEAD_INFO = {NUM_DUT[5:0], head_array[SEL]};

    assign dout[0] = 'd0123;
    assign head_array[0] = {4'd0, 6'd0, 6'd1, 5'd0, BITWIDTH_DATA[4:0]};
    assign rdy_filt[0] = 1'd0; 

    SKELETON_ECHO#(
        .BITWIDTH_SYS(BITWIDTH_DATA),
        .BITWIDTH_HEAD(BITWIDTH_HEAD)
    ) DUT_01 (
        .CLK_SYS(CLK),
        .RSTN(RSTN),
        .EN(SEL == 'd1),
        .TRGG_START_CALC(START_FLAG),
        .DATA_IN(DATA_IN),
        .DATA_OUT(dout[1]),
        .DATA_HEAD(head_array[1]),
        .DATA_VALID(rdy_filt[1])
    );

endmodule
