@echo off

pushd gold
..\bin\texturec.exe astc color gold_albedo_astc.tex gold_albedo.png
..\bin\texturec.exe astc normal gold_normal_astc.tex gold_normal.png
..\bin\texturec.exe astc mrao gold_mrao_astc.tex gold_metallic.png gold_roughness.png gold_ao.png
..\bin\texturec.exe bc color gold_albedo_bc.tex gold_albedo.png
..\bin\texturec.exe bc normal gold_normal_bc.tex gold_normal.png
..\bin\texturec.exe bc mrao gold_mrao_bc.tex gold_metallic.png gold_roughness.png gold_ao.png
popd

pushd grass
..\bin\texturec.exe astc color grass_albedo_astc.tex grass_albedo.png
..\bin\texturec.exe astc normal grass_normal_astc.tex grass_normal.png
..\bin\texturec.exe astc mrao grass_mrao_astc.tex grass_metallic.png grass_roughness.png grass_ao.png
..\bin\texturec.exe bc color grass_albedo_bc.tex grass_albedo.png
..\bin\texturec.exe bc normal grass_normal_bc.tex grass_normal.png
..\bin\texturec.exe bc mrao grass_mrao_bc.tex grass_metallic.png grass_roughness.png grass_ao.png
popd

pushd plastic
..\bin\texturec.exe astc color plastic_albedo_astc.tex plastic_albedo.png
..\bin\texturec.exe astc normal plastic_normal_astc.tex plastic_normal.png
..\bin\texturec.exe astc mrao plastic_mrao_astc.tex plastic_metallic.png plastic_roughness.png plastic_ao.png
..\bin\texturec.exe bc color plastic_albedo_bc.tex plastic_albedo.png
..\bin\texturec.exe bc normal plastic_normal_bc.tex plastic_normal.png
..\bin\texturec.exe bc mrao plastic_mrao_bc.tex plastic_metallic.png plastic_roughness.png plastic_ao.png
popd

pushd rusted_iron
..\bin\texturec.exe astc color rusted_iron_albedo_astc.tex rusted_iron_albedo.png
..\bin\texturec.exe astc normal rusted_iron_normal_astc.tex rusted_iron_normal.png
..\bin\texturec.exe astc mrao rusted_iron_mrao_astc.tex rusted_iron_metallic.png rusted_iron_roughness.png rusted_iron_ao.png
..\bin\texturec.exe bc color rusted_iron_albedo_bc.tex rusted_iron_albedo.png
..\bin\texturec.exe bc normal rusted_iron_normal_bc.tex rusted_iron_normal.png
..\bin\texturec.exe bc mrao rusted_iron_mrao_bc.tex rusted_iron_metallic.png rusted_iron_roughness.png rusted_iron_ao.png
popd

pushd wall
..\bin\texturec.exe astc color wall_albedo_astc.tex wall_albedo.png
..\bin\texturec.exe astc normal wall_normal_astc.tex wall_normal.png
..\bin\texturec.exe astc mrao wall_mrao_astc.tex wall_metallic.png wall_roughness.png wall_ao.png
..\bin\texturec.exe bc color wall_albedo_bc.tex wall_albedo.png
..\bin\texturec.exe bc normal wall_normal_bc.tex wall_normal.png
..\bin\texturec.exe bc mrao wall_mrao_bc.tex wall_metallic.png wall_roughness.png wall_ao.png
popd

pause