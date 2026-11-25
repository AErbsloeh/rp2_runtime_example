module DATA_REGISTER#(
    parameter integer BITWIDTH = 8
)(
    input wire CLK,
    input wire RSTN,
    input wire [BITWIDTH-1:0] DATA_IN,
    output reg [BITWIDTH-1:0] DATA_OUT
);

    always @(posedge CLK) begin
        if (!RSTN) begin
            DATA_OUT <= 'd0;
        end else begin
            DATA_OUT <= DATA_IN;
        end
    end
endmodule
