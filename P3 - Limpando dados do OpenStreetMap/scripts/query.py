import csv, sqlite3, pprint

def number_of_nodes():
    result = cur.execute('SELECT COUNT(*) FROM nodes')
    return result.fetchone()[0]


def number_of_ways():
    result = cur.execute('SELECT COUNT(*) FROM ways')
    return result.fetchone()[0]


def number_of_unique_users():
    result = cur.execute("""SELECT COUNT(DISTINCT(e.uid))
                             FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) e""")
    return result.fetchone()[0]


def top_contributing_users():
    users = []
    for row in cur.execute("""SELECT e.user, COUNT(*) as num 
            FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e 
            GROUP BY e.user 
            ORDER BY num DESC 
            LIMIT 5"""):
        users.append(row)
    return users


def amenities():
    amenities=[]
    for row in cur.execute("""SELECT value, COUNT(*) as num
                                FROM nodes_tags
                               WHERE key='amenity'
                               GROUP BY value
                               ORDER BY num DESC
                               LIMIT 10;"""):
        amenities.append(row)
    return amenities


def biggest_religion():
    for row in cur.execute("""SELECT nodes_tags.value, COUNT(*) as num 
            FROM nodes_tags 
                JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value="place_of_worship") i 
                ON nodes_tags.id=i.id 
            WHERE nodes_tags.key="religion" 
            GROUP BY nodes_tags.value 
            ORDER BY num DESC 
            LIMIT 1;"""):
        return row

def religion():
    religion = []
    for row in cur.execute("""SELECT nodes_tags.value, COUNT(*) as num 
            FROM nodes_tags 
                JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value="place_of_worship") i 
                ON nodes_tags.id=i.id 
            WHERE nodes_tags.key="religion" 
            GROUP BY nodes_tags.value 
            ORDER BY num DESC"""):
        religion.append(row)
    return religion

def popular_cuisines():
    for row in cur.execute("""SELECT nodes_tags.value, COUNT(*) as num 
            FROM nodes_tags 
                JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value="restaurant") i 
                ON nodes_tags.id=i.id 
            WHERE nodes_tags.key="cuisine" 
            GROUP BY nodes_tags.value 
            ORDER BY num DESC"""):
        return row

def cuisines():
    cuisines = []
    for row in cur.execute("""SELECT nodes_tags.value, COUNT(*) as num 
            FROM nodes_tags 
                JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value="restaurant") i 
                ON nodes_tags.id=i.id 
            WHERE nodes_tags.key="cuisine" 
            GROUP BY nodes_tags.value 
            ORDER BY num DESC
            LIMIT 5"""):
        cuisines.append(row)
    return cuisines

if __name__ == '__main__':
    con = sqlite3.connect("curitiba.db")
    cur = con.cursor()

    print("Number of nodes: {}".format(number_of_nodes()))
    print("\nNumber of ways: {}".format(number_of_ways()))
    print("\nNumber of unique users: {}".format(number_of_unique_users()))
    print("\nTop 5 contributing users: ")
    pprint.pprint(top_contributing_users())
    print("\nTop 10 Amenities: {}".format(amenities()))
    print("\nReligion:")
    pprint.pprint(religion())
    print("\nBiggest religion: {}".format(biggest_religion()))
    print("\nTop 5 Cuisines:")
    pprint.pprint(cuisines())

