
from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheTypes
# for sqlite
GutenbergCache.create()

#for mongodb
# GutenbergCache.create(type=GutenbergCacheTypes.CACHE_TYPE_MONGODB)

if __name__ == "__main__":

    cache  = GutenbergCache.get_cache()

    # Retrieves all 
    results = [link for link in cache.native_query("Select * From \
                                                    BOOKS AS B \
                                                    LEFT JOIN TITLES AS T \
                                                    ON T.bookid=B.id"
                                                   )]


    print('End Run')