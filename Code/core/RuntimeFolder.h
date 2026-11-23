#ifndef HEADER_RuntimeFolder
#define HEADER_RuntimeFolder

#include <wx/string.h>

//! Locates the folder holding SwisTrack's runtime files (the "Components" folder and "default.swistrack").
/*!
	The following locations are searched in order, and the first one containing a "Components" folder is returned:
	 - the environment variable SWISTRACK_HOME
	 - the current working directory (classic layout: run from SwisTrackEnvironment)
	 - the folder of the executable (Windows, build trees)
	 - ../Resources relative to the executable (macOS application bundle)
	 - ../share/swistrack relative to the executable (Linux/Unix installs)
	If none of them contains a "Components" folder, the current working directory is returned.
*/
wxString FindRuntimeFolder();

#endif
