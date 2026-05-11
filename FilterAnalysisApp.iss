[Setup]
AppName=Filter Analysis App
AppVersion=1.0
AppPublisher=Ali Peykar
AppCopyright=Copyright (C) 2026 Ali Peykar
DefaultDirName={autopf}\FilterAnalysisApp
DefaultGroupName=Filter Analysis App
OutputDir=Output
OutputBaseFilename=FilterAnalysisApp_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=admin
WizardSmallImageFile=wizard_small.bmp

[Tasks]
Name: desktopicon; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
Source: "dist\FilterAnalysisApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Filter Analysis App"; Filename: "{app}\FilterAnalysisApp.exe"
Name: "{commondesktop}\Filter Analysis App"; Filename: "{app}\FilterAnalysisApp.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FilterAnalysisApp.exe"; Description: "Launch Filter Analysis App"; Flags: nowait postinstall skipifsilent
