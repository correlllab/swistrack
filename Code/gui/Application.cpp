#include "Application.h"
#define THISCLASS Application
#include <wx/app.h>
#include <wx/image.h>
#include <wx/filefn.h>

#include "ConfigurationReaderXML.h"
#include "SwisTrackCore.h"
#include "Random.h"
#include "RuntimeFolder.h"

IMPLEMENT_APP(Application)

bool THISCLASS::OnInit() {
	// Set some main application parameters.
	SetVendorName(wxT("SwisTrack Community"));
	SetAppName(wxT("SwisTrack"));

	// Locate the folder with the component descriptions and the default configuration
	mApplicationFolder = FindRuntimeFolder();

	// Initialize the random number generator
	Random::Initialize();

	// Initialize all available image handlers
	wxInitAllImageHandlers();

	// Create the main application window
	mSwisTrack = new SwisTrack(_T("SwisTrack"), wxPoint(-1, -1), wxSize(1600, 1200));

	// Open a file
	if (argc > 1) {
		mSwisTrack->OpenFile(wxFileName(argv[1]), true, false);
	} else {
		mSwisTrack->NewFile();
	}

	// Show
	mSwisTrack->Show(true);
	SetTopWindow(mSwisTrack);

#if defined(__WIN16__) || defined(__WXMOTIF__)
	int width, height;
	frame->GetSize(&width, &height);
	frame->SetSize(-1, -1, width, height);
#endif

	return true;
}

int THISCLASS::OnExit() {
	// Uninitialize the random number generator
	Random::Uninitialize();

	return 0;
}
