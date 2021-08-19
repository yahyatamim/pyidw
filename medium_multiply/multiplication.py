class Multiplication:
    """
    Instantiate a multiplication operation.
    Numbers will be multiplied by the given multiplier. Md.Yahya Tamim.
    
    :param multiplier: The multiplier.
    :type multiplier: int
    """
    
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def multiply(self, number):
        """
        Multiply a given number by the multiplier.
        
        :param number: The number to multiply.
        :type number: int
    
        :return: The result of the multiplication.
        :rtype: int
        """
        
        return number * self.multiplier

# Instantiate a Multiplication object
multiplication = Multiplication(2)

# Call the multiply method
print(multiplication.multiply(5))
