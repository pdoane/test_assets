@echo off

pushd gold
..\bin\texturec.exe color gold_albedo.tex gold_albedo.png
..\bin\texturec.exe normal gold_normal.tex gold_normal.png
..\bin\texturec.exe mrao gold_mrao.tex gold_metallic.png gold_roughness.png gold_ao.png
popd

pushd grass
..\bin\texturec.exe color grass_albedo.tex grass_albedo.png
..\bin\texturec.exe normal grass_normal.tex grass_normal.png
..\bin\texturec.exe mrao grass_mrao.tex grass_metallic.png grass_roughness.png grass_ao.png
popd

pushd plastic
..\bin\texturec.exe color plastic_albedo.tex plastic_albedo.png
..\bin\texturec.exe normal plastic_normal.tex plastic_normal.png
..\bin\texturec.exe mrao plastic_mrao.tex plastic_metallic.png plastic_roughness.png plastic_ao.png
popd

pushd rusted_iron
..\bin\texturec.exe color rusted_iron_albedo.tex rusted_iron_albedo.png
..\bin\texturec.exe normal rusted_iron_normal.tex rusted_iron_normal.png
..\bin\texturec.exe mrao rusted_iron_mrao.tex rusted_iron_metallic.png rusted_iron_roughness.png rusted_iron_ao.png
popd

pushd wall
..\bin\texturec.exe color wall_albedo.tex wall_albedo.png
..\bin\texturec.exe normal wall_normal.tex wall_normal.png
..\bin\texturec.exe mrao wall_mrao.tex wall_metallic.png wall_roughness.png wall_ao.png
popd

pause