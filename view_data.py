def view_data(areaFile):

	from subprocess import call
	fName = areaFile.file.name
	status = call("Data_Viewer " + fName, shell=True)
	return status
