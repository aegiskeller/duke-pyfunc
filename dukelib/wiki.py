import wikipedia
import yake

# build a function to get the summary page for a city
def city_summary(city):
    """
    This function returns the summary page for a city.
    """
    return wikipedia.summary(city)  # return the summary page for the city

# a function that takes the summary and finds the keywords using yake
def city_keywords(city):
    """
    This function returns the keywords for a city.
    """
    content = wikipedia.summary(city)
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(content)
    # return a dicrtionary of the top 10 keywords   
    return {keyword: score for keyword, score in keywords[:10]}