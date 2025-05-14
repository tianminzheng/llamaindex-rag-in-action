from pydantic import BaseModel


class Restaurant(BaseModel):

    restaurant: str
    food: str
    discount: str
    price: str
    rating: str
    review: str
