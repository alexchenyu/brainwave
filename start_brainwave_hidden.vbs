Set WshShell = CreateObject("WScript.Shell")
' Get the directory of this VBS script
Dim scriptDir
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
' Run the batch file in the same directory, hidden (0 = hide window)
WshShell.Run scriptDir & "\start_brainwave.bat", 0, False
Set WshShell = Nothing
