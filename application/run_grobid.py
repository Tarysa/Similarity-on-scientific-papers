import os
import sys
current_path = os.path.dirname(__file__)

# import grobid_client as grobid
from grobid_client_python.grobid_client.grobid_client import GrobidClient
# parameters to set for the process method
# all possible services
servicesList = ["processFulltextDocument", "processHeaderDocument", "processReferences"]
# the service we use
service = servicesList[0]
# where there raw files are
input_path = os.path.join(current_path, 'static', 'uploaded_file', 'temp_pdf')
# where the output should be placed
output_path = os.path.join(current_path, 'static', 'uploaded_file', 'grobid')
# concurrency for service usage
n = 20
# generate random xml:id to textual XML elements of the result files
generateIDs = True
# call GROBID with consolidation of the metadata extracted from the header
consolidate_header = True
# call GROBID with consolidation of the extracted bibliographical references
consolidate_citations = True
# The extraction of raw citations
include_raw_citations = True
# the extraciton of raw affiliations
include_raw_affiliations = True
# add the original PDF coordinates (bounding boxes) to the extracted elements
teiCoordinates = True
# force re-processing pdf input files when tei output files already exist
force = True
# print information about processed files in the console
verbose = True

def runGrobid():
# the config lies in the root directory
    current_path = os.path.dirname(__file__)
    client = GrobidClient(config_path=os.path.join(current_path, 'config.json'))
    client.process(service, input_path, output=output_path, n=n, generateIDs=generateIDs,
    consolidate_header=consolidate_header, consolidate_citations=consolidate_citations,
    include_raw_citations=include_raw_citations, include_raw_affiliations=include_raw_affiliations,
    teiCoordinates=teiCoordinates, force=force, verbose=verbose)