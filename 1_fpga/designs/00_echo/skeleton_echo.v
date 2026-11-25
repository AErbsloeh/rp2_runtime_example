//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     20.12.2024 12:54:02
// Copied on: 	    {$date_copy_created}
// Module Name:     SKELETON_ECHO
// Target Devices:  FPGA
// Tool Versions:   1v0
// Description:     Skeleton for doing echo test on device
// Dependencies:    None
//
// State: 	        Works! (System Test done: 16.01.2025)
// Improvements:    None
// Parameters:      BITWIDTH_SYS --> Bitwidth of data bus on device
//                  BITWIDTH_HEAD --> Bitwidth of metadata (skeleton properties)
//////////////////////////////////////////////////////////////////////////////////

module SKELETON_ECHO#(
    parameter integer BITWIDTH_SYS = 16,
    parameter integer BITWIDTH_HEAD = 26
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire TRGG_START_CALC,
    input wire [BITWIDTH_SYS-'d1:0] DATA_IN,
    output reg [BITWIDTH_SYS-'d1:0] DATA_OUT,
    output wire [BITWIDTH_HEAD-'d1:0] DATA_HEAD,
    output wire DATA_VALID
);

    reg first_run_done;
    assign DATA_HEAD = {4'd1, 6'd1, 6'd1, BITWIDTH_SYS[4:0], BITWIDTH_SYS[4:0]};
    assign DATA_VALID = first_run_done && !TRGG_START_CALC;

    // --- Control lines
    always@(posedge CLK_SYS) begin
        if(!(RSTN && EN)) begin
            first_run_done <= 1'd0;
            DATA_OUT <= 'd0;
        end else begin
            first_run_done <= (TRGG_START_CALC) ? 1'd1 : first_run_done;
            DATA_OUT <= (TRGG_START_CALC) ? DATA_IN : DATA_OUT;
        end
    end
endmodule
