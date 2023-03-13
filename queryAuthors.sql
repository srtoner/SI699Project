-- Get Authors with the most books
SELECT a.authorid, a.name, count(a.bookid) as book_count from 
    (SELECT * from authors
    LEFT JOIN book_authors
    ON id = authorid) as a
LEFT JOIN books as b
ON b.id = a.bookid
GROUP BY a.id, a.name
ORDER BY book_count DESC
LIMIT 100;