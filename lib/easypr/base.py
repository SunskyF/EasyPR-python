class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance


class Plate(object):
    def __init__(self):
        self.bColored = True

    @property
    def plate_image(self):
        return self.__plate_image

    @plate_image.setter
    def plate_image(self, plate):
        self.__plate_image = plate

    @property
    def plate_pos(self):
        return self.__plate_pos

    @plate_pos.setter
    def plate_pos(self, param):
        self.__plate_pos = param

    @property
    def plate_str(self):
        return self.__plate_str

    @plate_str.setter
    def plate_str(self, param):
        self.__plate_str = param

    @property
    def plate_type(self):
        return self.__plate_type

    @plate_type.setter
    def plate_type(self, param):
        self.__plate_type = param
