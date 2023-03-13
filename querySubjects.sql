SELECT bs.subjectid, b.name, b.numdownloads FROM 
    (SELECT * FROM books 
     LEFT JOIN titles AS T
     ON T.bookid = books.id) AS b
LEFT JOIN book_subjects AS bs
ON b.id = bs.bookid and bs.subjectid=44
ORDER BY b.numdownloads DESC
LIMIT 25;


