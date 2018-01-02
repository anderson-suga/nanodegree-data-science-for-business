import xml.etree.cElementTree as ET
import pprint
import re

DATAFILE = "curitiba.osm"

zip_code_correct = re.compile(r'\d{5}-\d{3}')
zip_code_only_digits = re.compile(r'\d{8}')
zip_code_dot = re.compile(r'\d{2}.\d{3}-\d{3}')

def is_zip_code(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_zipcode(zip_code, code):
    if not zip_code_correct.match(code):
        zip_code.add(code)

def process_zipcode():
    zip_code = set()

    for event, elem in ET.iterparse(DATAFILE):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_zip_code(tag):
                    audit_zipcode(zip_code, tag.attrib['v'])

    return zip_code

def udpate_zipcode(zipcode_error):
    if zipcode_error[0] != '8':
        return None
    elif zip_code_only_digits.match(zipcode_error):
        return zipcode_error[0:5]+'-'+zipcode_error[5:8]
    elif zip_code_dot.match(zipcode_error):
        return zipcode_error.replace('.','')


if __name__ == "__main__":
    zip_code = process_zipcode()

    pprint.pprint(zip_code)

    print('\nUpdate - Zip Code:')

    for zipcode_error in zip_code:
        zipcode_correct = udpate_zipcode(zipcode_error)
        print("{} => {}".format(zipcode_error,zipcode_correct))
