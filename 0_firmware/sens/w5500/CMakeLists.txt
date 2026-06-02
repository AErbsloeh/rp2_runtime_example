message(STATUS "peri_w5500 available at: ${CMAKE_CURRENT_LIST_FILE}")
add_library(peri_w5500 INTERFACE)

# --- Fetch external libs from GitHub ---
if(PICO_W5500_ENABLED)
    include(FetchContent)
    FetchContent_Declare(
        wiznet_eth
        GIT_REPOSITORY https://github.com/Wiznet/ioLibrary_Driver.git
        GIT_TAG master  
    )
    FetchContent_MakeAvailable(wiznet_eth)
    #add_subdirectory(${CMAKE_BINARY_DIR}/_dep/wiznet_eth-src/Ethernet build)
    message(STATUS "download code for W5500 Ethernet support")
    add_definitions(-D_WIZCHIP_=W5500)

    target_sources(peri_w5500 INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/w5500_udp.c
        ${CMAKE_CURRENT_SOURCE_DIR}/wizchip_spi.c
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Application/loopback/loopback.c
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Application/multicast/multicast.c
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Ethernet/socket.c
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Ethernet/wizchip_conf.c
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Ethernet/W5500/w5500.c
    )
    target_link_libraries(peri_w5500 INTERFACE
        pico_stdlib
        hardware_gpio
        hal_spi
        hardware_dma
    )

    message(STATUS "${CMAKE_SOURCE_DIR}")

    # --- Link to project ---
    target_include_directories(peri_w5500 INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Ethernet
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Ethernet/W5500
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Application/multicast
        ${CMAKE_SOURCE_DIR}/build/_deps/wiznet_eth-src/Application/loopback
    )
else()
    target_include_directories(peri_w5500 INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
    )
endif()
