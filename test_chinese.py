from mimesis import Generic
from mimesis.locales import Locale

# Test Chinese text handling
def test_chinese():
    generic = Generic(locale=Locale.ZH)
    name = generic.person.full_name()
    address = generic.address.address()
    print(f"Generated Chinese name: {name}")
    print(f"Generated Chinese address: {address}")

if __name__ == "__main__":
    test_chinese()
