# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.1)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.1.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2_w CACHE STRING "Board type")
set(PROGRAM_VERSION "0.1")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)
project(main C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name
add_executable(main 
    main.c 
    callbacks/gpio_callbacks.c
    callbacks/rpc_callbacks.c
    src/init_system.c
    src/testbench.c
    hardware_io.c
)

pico_set_program_name(main "main")
pico_set_program_version(main "${PROGRAM_VERSION}")
configure_file(version.h.in version.h @ONLY)
target_include_directories(main PRIVATE ${CMAKE_BINARY_DIR})

# Adding custom device/HAL library
include(FetchContent)
FetchContent_Declare(
    custom_library
    GIT_REPOSITORY https://github.com/AErbsloeh/rp2_runtime_library.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(custom_library)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../rp2_runtime_library custom_library)

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(main false)
pico_enable_stdio_usb(main true)

# Add the standard library to the build
target_link_libraries(main
    pico_stdlib
    pico_bootsel_via_double_reset
    hardware_watchdog
    hardware_clocks
    hardware_gpio
    # ---- Include Hardware Abstraction Layer (HAL) libraries
	hal_adc
    #hal_btspp
    hal_clk
    hal_daq
    #hal_i2c
    hal_led
    #hal_osc
    #hal_pwm
    #hal_pwr_single
    #hal_pwr_dual
    hal_spi
    hal_tmr
    #hal_uart
    hal_usb
    #hal_wifi_tcp
    #hal_wifi_udp
     # ---- Include ADC libraries
    #adc_ad4858
    #adc_ad7779
    #adc_ads8881
    # ---- Include DAC libraries
    #dac_ad57x4
    #dac_ad5765
    #dac_ltc2668
    # ---- Include peripherie libraries
    #peri_sd
    #peri_w5500
    #peri_ws2812
    #peri_oled4440
    #peri_fpga_flash
    #peri_fpga_spi
    # ---- Include digital potentiometer libraries
    #poti_ad526x   
    #poti_ad528x
    #poti_ad5141_i2c
    #poti_ad5141_spi
    #poti_ad5142a
    # ---- Include sensor libraries
    #sens_adxl345_i2c
    #sens_bmi270_i2c
    #sens_pac193x
    #sens_sht21
    #sens_veml7700
    #sens_vl6180
)
get_target_property(LINK_LIBS main LINK_LIBRARIES)

# Define definition and set flags if used (necessary for external file handling)
if(PICO_BOARD MATCHES "_w")
    target_compile_definitions(main PUBLIC ADD_CYW43_SUPPORT)
endif()

if("hal_btspp" IN_LIST LINK_LIBS)
    set(PICO_BTSTACK_ENABLED ON CACHE BOOL "Enable Bluetooth SPP HAL library" FORCE)
else()
    set(PICO_BTSTACK_ENABLED OFF CACHE BOOL "Disable Bluetooth SPP HAL library" FORCE)
endif()

if("hal_wifi_tcp" IN_LIST LINK_LIBS)
    set(PICO_WIFI_TCP_ENABLED ON CACHE BOOL "Enable Wifi TCP HAL library" FORCE)
else()
    set(PICO_WIFI_TCP_ENABLED OFF CACHE BOOL "Disable Wifi TCP HAL library" FORCE)
endif()

if("hal_wifi_udp" IN_LIST LINK_LIBS)
    set(PICO_WIFI_UDP_ENABLED ON CACHE BOOL "Enable Wifi UDP HAL library" FORCE)
else()
    set(PICO_WIFI_UDP_ENABLED OFF CACHE BOOL "Disable Wifi UDP HAL library" FORCE)
endif()

if("peri_sd" IN_LIST LINK_LIBS)
    set(PICO_SD_ENABLED ON CACHE BOOL "Enable SD card library" FORCE)
else()
    set(PICO_SD_ENABLED OFF CACHE BOOL "Disable SD card library" FORCE)
endif()

if("peri_w5500" IN_LIST LINK_LIBS)
    set(PICO_W5500_ENABLED ON CACHE BOOL "Enable Ethernet W5500 library" FORCE)
    add_definitions(-D_WIZCHIP_=W5500)
else()
    set(PICO_W5500_ENABLED OFF CACHE BOOL "Disable Ethernet W5500 library")
endif()

# Add the standard include files to the build
target_include_directories(main PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
pico_add_extra_outputs(main)
