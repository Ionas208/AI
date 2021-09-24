class Person:
    def __init__(self, name, address, contact):
        print('alive')
        self.name = name
        self.address = address
        self.contact = contact

    def get_name(self):
        return self.name
    
    def get_address(self):
        return self.address
    
    def get_contact(self):
        return self.contact


class Address:
    def __init__(self, street, number, postcode, city, country):
        self.street = street
        self.postcode = postcode
        self.city = city
        self.country = country
    
    def get_street(self):
        return self.street
    
    def get_number(self):
        return self.number
    
    def get_postcode(self):
        return self.postcode
    
    def get_city(self):
        return self.city
    
    def get_country(self):
        return self.country


class Contact:
    def __init__(self, email):
        self.email = email

    def get_email(self):
        return self.email
    

contact = Contact('test@test.com')
address = Address('Teststreet', 1, 1234, 'Testcity', 'Testcountry')
person = Person('Franz', address, contact)