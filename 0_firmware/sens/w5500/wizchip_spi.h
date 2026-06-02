/**
    Copyright (c) 2022 WIZnet Co.,Ltd

    SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef _WIZCHIP_SPI_H_
#define _WIZCHIP_SPI_H_

/**
    ----------------------------------------------------------------------------------------------------
    Macros
    ----------------------------------------------------------------------------------------------------
*/
/**
    ----------------------------------------------------------------------------------------------------
    Functions
    ----------------------------------------------------------------------------------------------------
*/
/* wizchip */
/*! \brief Set CS pin
    \ingroup wizchip_spi

    Set chip select pin of spi0 to low(Active low).

    \param none
*/

/*! \brief Enter a critical section
    \ingroup wizchip_spi

    Set ciritical section enter blocking function.
    If the spin lock associated with this critical section is in use, then this
    method will block until it is released.

    \param none
*/
static void wizchip_critical_section_lock(void);

/*! \brief Release a critical section
    \ingroup wizchip_spi

    Set ciritical section exit function.
    Release a critical section.

    \param none
*/
static void wizchip_critical_section_unlock(void);


/*! \brief Initialize a critical section structure
    \ingroup wizchip_spi

    The critical section is initialized ready for use.
    Registers callback function for critical section for WIZchip.

    \param none
*/
void wizchip_cris_initialize(void);


/* Network */
/*! \brief Initialize network
    \ingroup wizchip_spi

    Set network information.

    \param net_info network information.
*/
void network_initialize(wiz_NetInfo net_info);


/*! \brief Print network information
    \ingroup wizchip_spi

    Print network information about MAC address, IP address, Subnet mask, Gateway, DHCP and DNS address.

    \param net_info network information.
*/
void print_network_information(wiz_NetInfo net_info);


/*! \brief Print IPv6 Address
    \ingroup wizchip_spi

    Print IPv6 Address.

    \param net_info network information.
*/
void print_ipv6_addr(uint8_t* name, uint8_t* ip6addr);


#endif /* _WIZCHIP_SPI_H_ */
