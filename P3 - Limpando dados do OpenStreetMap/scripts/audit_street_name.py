import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE_sample = "curitiba.osm"
regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

# expected names in the dataset

expected = ["Rua","Avenida","Alameda","Praça","Rodovia", "Travessa","Estrada", "Largo", "BR-116", "Linha", "Marginal", "Acesso","Centro"]

mapping = {"Av": "Avenida",
           "Av.": "Avenida",
           "R": "Rua",
           "R.": "Rua",
           "RUA": "Rua",
           "Ana": "Rua Ana",
           "Angelo Francisco Borato": "Rua Ângelo Francisco Borato",
           "Comendador Franco": "Avenida Comendador Franco",
           "Residencial": "Rua",
           "Vicente Za": "Rua Carlos Vicente Zapxon",
           "Domingos Benatto": "Rua Domingos Benatto",
           "José Maria Da Silva Paramos 551": "Rua Domingos Jorge Velho",
           "Rue Desembargador Motta - 2311 - Batel": "Rua Desembargador Motta",
           "Rua Samanbaia -179- São Francisco -Araucária": "Rua Samambaia"
}

# Search string for the regex. If it is matched and not in the expected list then add this as a key to the set.
def audit_street(street_types, street_name):
    m = regex.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


# Check if it is a street name
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

# return the list that satify the above two functions
def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, tag.attrib['v'])

    return street_types

# change string into titleCase except for UpperCase
def string_case(s):
    if s.isupper():
        return s
    else:
        return s.title()


# return the updated names
def update_name(name, mapping):

    name = name.split(' ')
    for i in range(len(name)):
        if name[i] in mapping:
            name[i] = mapping[name[i]]
            name[i] = string_case(name[i])
        else:
            name[i] = string_case(name[i])

    name = ' '.join(name)

    if name in mapping:
        return mapping[name]

    return name

if __name__ == "__main__":
    update_street = audit(OSMFILE_sample)

    # print the existing names
    pprint.pprint(dict(update_street))

    print('\nUpdate - Street Name:')

    # print the updated names
    for street_type, ways in update_street.items():
        for name in ways:
            better_name = update_name(name, mapping)
            print ("{} => {}".format(name,better_name.title()))