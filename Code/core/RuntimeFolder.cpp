#include "RuntimeFolder.h"

#include <wx/filefn.h>
#include <wx/filename.h>
#include <wx/stdpaths.h>
#include <wx/utils.h>

static bool HasComponentsFolder(const wxString & folder) {
	if (folder.IsEmpty()) return false;
	wxFileName components(folder, wxT(""));
	components.AppendDir(wxT("Components"));
	return components.DirExists();
}

wxString FindRuntimeFolder() {
	wxArrayString candidates;

	wxString env;
	if (wxGetEnv(wxT("SWISTRACK_HOME"), &env)) candidates.Add(env);

	candidates.Add(wxGetCwd());

	wxFileName executable(wxStandardPaths::Get().GetExecutablePath());
	executable.Normalize(wxPATH_NORM_ABSOLUTE | wxPATH_NORM_DOTS);
	wxString executableFolder = executable.GetPath();
	candidates.Add(executableFolder);

	wxFileName resources(executableFolder, wxT(""));
	resources.RemoveLastDir();
	resources.AppendDir(wxT("Resources"));
	candidates.Add(resources.GetPath());

	wxFileName share(executableFolder, wxT(""));
	share.RemoveLastDir();
	share.AppendDir(wxT("share"));
	share.AppendDir(wxT("swistrack"));
	candidates.Add(share.GetPath());

	for (const wxString & candidate : candidates)
		if (HasComponentsFolder(candidate)) return candidate;

	return wxGetCwd();
}
