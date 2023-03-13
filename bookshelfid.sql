-- Not useful; bookshelveid is not a well maintained field
SELECT * FROM (
    SELECT bs.id, bs.name, COUNT(b.id) as BOOK_COUNT FROM 
    (SELECT DISTINCT id, name FROM bookshelves) as bs        
        RIGHT JOIN 
    (SELECT id, bookshelveid, languageid, typeid FROM books
        WHERE languageid=1) as b
        ON bs.id=b.bookshelveid) as bbs
GROUP BY bbs.id, bbs.name
ORDER BY BOOK_COUNT DESC;
