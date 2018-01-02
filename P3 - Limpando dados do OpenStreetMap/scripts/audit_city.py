import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE_sample = "curitiba.osm"

regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

# expected names in the dataset
expected = ["Curitiba","São José dos Pinhais","Pinhais","Colombo","Campo","Araucária","Almirante", "Rio"]

mapping = {'Alto da Rua XV' : "Curitiba",
           'Araucaria' : "Araucária",
           'Batel' : "Curitiba",
           'Cajuru' : "Curitiba",
           'Curiiba' : "Curitiba",
           'Curitba' : "Curitiba",
           'Fanny' : "Curitiba",
           'Jardim Amélia - Pinhais': "Pinhais",
           'Rebouças' : "Curitiba",
           'Sao José dos Pinhais': "São José dos Pinhais",
           'São José dos Pinhais - PR': "São José dos Pinhais",
           'São José dos Pinias': "São José dos Pinhais",
           "Tarumã" : "Curitiba",
           'curitiba' : "Curitiba"
}

# Search string for the regex. If it is matched and not in the expected list then add this as a key to the set.
def audit_city(city_types, city_name):
    m = regex.search(city_name)
    if m:
        city_type = m.group()
        if city_type not in expected:
            city_types[city_type].add(city_name)


# Check if it is a street name
def is_city_name(elem):
    return (elem.attrib['k'] == "addr:city")

# return the list that satify the above two functions
def audit():
    osm_file = open(OSMFILE_sample, "r")
    city_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_city_name(tag):
                    audit_city(city_types, tag.attrib['v'])

    return city_types

# change string into titleCase except for UpperCase
def string_case(s):
    if s.isupper():
        return s
    else:
        return s.title()


# return the updated names
def update_name(name, mapping):

    if name in mapping:
        return mapping[name]

    return name

if __name__ == "__main__":
    update_city = audit()
    pprint.pprint(dict(update_city))

    print('\nUpdate - City:')

    # print the updated names
    for city_type, ways in update_city.items():
        for name in ways:
            better_name = update_name(name, mapping)
            print ("{} => {}".format(name,better_name))
