SELECT bs.subjectid, s.name, count(bs.bookid) as book_count from book_subjects as bs
LEFT JOIN subjects as s
ON s.id = bs.subjectid
GROUP BY s.id, s.name
ORDER BY book_count DESC
LIMIT 25;


-- 44|PS|11113 American and Canadian Literature
-- 10|PR|10009 English Literature
-- 13|PZ|7282 Children's Stories
-- 16|PQ|4843 Romance Languages
-- 28|PT|2956 Germanic Languages
-- 80|Science fiction|2938
-- 14|Short stories|2748
-- 2|AP|2583 General Works, Periodicals
-- 71|Fiction|1979
-- 46|DA|1663 History: General and Eastern Hemisphere
-- 105|Adventure stories|1475
-- 146|PH|1379 Hungarian & Finnish
-- 18|PN|992 General, Criticism, and Collections
-- 79|DC|961 History, France, Andorra, Monaco
-- 308|Historical fiction|947
-- 91|DS|943 History: Asia
-- 30|Conduct of life -- Juvenile fiction|876
-- 158|Love stories|858
-- 291|D501|827 History: WWI
-- 192|Detective and mystery stories|818
-- 51|Man-woman relationships -- Fiction|809
-- 197|BX|745 Philosophy / Religion: Christianity
-- 29|Poetry|681 
-- 68|QH|670 Natural History