#include "ComponentOutputTracksCSV.h"
#define THISCLASS ComponentOutputTracksCSV

#include <wx/filename.h>

THISCLASS::ComponentOutputTracksCSV(SwisTrackCore * stc):
	Component(stc, wxT("OutputTracksCSV")),
	mFile() {

	// Data structure relations
	mCategory = &(mCore->mCategoryOutput);
	AddDataStructureRead(&(mCore->mDataStructureInput));
	AddDataStructureRead(&(mCore->mDataStructureTracks));

	// Read the XML configuration file
	Initialize();
}

THISCLASS::~ComponentOutputTracksCSV() {
}

void THISCLASS::OnStart() {
	wxString filenameString = GetConfigurationString(wxT("File"), wxT("tracks.csv"));
	if (filenameString.IsEmpty()) filenameString = wxT("tracks.csv");

	wxFileName filename = mCore->GetRunFileName(filenameString);
	if (! filename.IsOk()) {
		AddError(wxT("Invalid output file name (save the configuration first)."));
		return;
	}

	filename.Mkdir(0777, wxPATH_MKDIR_FULL);
	mFile.open(filename.GetFullPath().ToStdString(), std::ios::out | std::ios::trunc);
	if (! mFile.is_open()) {
		AddError(wxT("Unable to open '") + filename.GetFullPath() + wxT("'."));
		return;
	}

	mFile.precision(9);
	mFile << "frame,id,x,y,vx,vy\n";
	mFile.flush();
}

void THISCLASS::OnReloadConfiguration() {
}

void THISCLASS::OnStep() {
	if (! mFile.is_open()) return;

	auto tracks = mCore->mDataStructureTracks.mTracks;
	if (! tracks) {
		AddWarning(wxT("No tracks."));
		return;
	}

	// One row per track that has a point in the current frame
	int frameNumber = mCore->mDataStructureInput.mFrameNumber;
	for (auto & entry : *tracks) {
		auto & track = entry.second;
		if (track.mTrajectory.empty()) continue;
		auto & point = track.mTrajectory.back();
		if (point.frameNumber != frameNumber) continue;
		mFile << frameNumber << ',' << track.mID << ',' << point.x << ',' << point.y << ',' << track.mVelocity.x << ',' << track.mVelocity.y << '\n';
	}

	// SwisTrack-batch exits without stopping the core when the input ends, so flush after every frame
	mFile.flush();
}

void THISCLASS::OnStepCleanup() {
}

void THISCLASS::OnStop() {
	if (! mFile.is_open()) return;
	mFile.flush();
	mFile.close();
}
