import csv
import operator
from configparser import ConfigParser
from pathlib import Path
import logging

def confirm_param_or_exit(paramfile, preset, PRESET_DICT) -> Path:
    if paramfile is None and preset is None:
        logging.error(f"param file not specified")
        raise RuntimeError(f"param file not specified")


    if preset is not None:
        _paramfile = PRESET_DICT.get(preset)
        if _paramfile is None or not _paramfile.exists():
            logging.error(f"predefined param file {_paramfile} does not exist yet.")
            #raise NotImplementedError(f"{_paramfile} does not exist")
    return _paramfile

def read_properties(filename, comment_chars=("#", ":")):
    """Reads a given properties file with each line of the format key=value.  Returns a dictionary containing the pairs.

    Keyword arguments:
        filename -- the name of the file to be read
    """
    result = {}
    with open(filename, "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter="=", escapechar="\\", quoting=csv.QUOTE_NONE
        )
        for row in reader:
            if not row:  # empty
                continue
            if any(row[0].strip().startswith(x) for x in comment_chars):
                continue
            if len(row) != 2:
                raise csv.Error("Too many fields on row with contents: " + str(row))
            result[row[0]] = row[1]
    return result


def write_properties(filename, dictionary):
    """Writes the provided dictionary in key-sorted order to a properties file with each line of the format key=value

    Keyword arguments:
        filename -- the name of the file to be written
        dictionary -- a dictionary containing the key/value pairs.
    """
    with open(filename, "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter="=", escapechar="\\", quoting=csv.QUOTE_NONE
        )
        for key, value in sorted(dictionary.items(), key=operator.itemgetter(0)):
            writer.writerow([key, value])


# def main():
#    data={
#        "Hello": "5+5=10",
#        "World": "Snausage",
#        "Awesome": "Possum"
#    }
#
#    filename="test.properties"
#    write_properties(filename,data)
#    newdata=read_properties(filename)
#
#    print "Read in: "
#    print newdata
#    print
#
#    contents=""
#    with open(filename, 'rb') as propfile:
#        contents=propfile.read()
#    print "File contents:"
#    print contents
#
#    print ["Failure!", "Success!"][data == newdata]
#    return

