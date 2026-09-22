# LEDs
set_property -dict {PACKAGE_PIN W13 IOSTANDARD LVCMOS33} [get_ports {led_tri_o[0]}]
set_property -dict {PACKAGE_PIN W16 IOSTANDARD LVCMOS33} [get_ports {led_tri_o[1]}]
set_property -dict {PACKAGE_PIN W15 IOSTANDARD LVCMOS33} [get_ports {led_tri_o[2]}]
set_property -dict {PACKAGE_PIN W11 IOSTANDARD LVCMOS33} [get_ports {led_tri_o[3]}]
set_property -dict {PACKAGE_PIN T11 IOSTANDARD LVCMOS33} [get_ports {led_tri_o[4]}]
set_property -dict {PACKAGE_PIN U9  IOSTANDARD LVCMOS33} [get_ports {led_tri_o[5]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {led_tri_o[6]}]
set_property -dict {PACKAGE_PIN U8  IOSTANDARD LVCMOS33} [get_ports {led_tri_o[7]}]

# Buttons
set_property -dict {PACKAGE_PIN Y14 IOSTANDARD LVCMOS33} [get_ports {btn_tri_i[0]}]
set_property -dict {PACKAGE_PIN W14 IOSTANDARD LVCMOS33} [get_ports {btn_tri_i[1]}]
set_property -dict {PACKAGE_PIN Y19 IOSTANDARD LVCMOS33} [get_ports {btn_tri_i[2]}]
set_property -dict {PACKAGE_PIN Y16 IOSTANDARD LVCMOS33} [get_ports {btn_tri_i[3]}]

# Switches
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[0]}]
set_property -dict {PACKAGE_PIN Y17 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[1]}]
set_property -dict {PACKAGE_PIN Y13 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[2]}]
set_property -dict {PACKAGE_PIN Y12 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[3]}]
set_property -dict {PACKAGE_PIN Y11 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[4]}]
set_property -dict {PACKAGE_PIN T12 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[5]}]
set_property -dict {PACKAGE_PIN T10 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[6]}]
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {sw_tri_i[7]}]