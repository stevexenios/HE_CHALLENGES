USE Hacker_Earth_DB_1
GO

SELECT E.Employee_ID, E.[Name]
from EMPLOYEE as E
INNER JOIN [SERVICE] S
ON E.Employee_ID = S.Employee_ID
WHERE E.Gender = 'F' AND S.Pay_Scale > 4.0 AND S.Time_since_promotion > 1
ORDER BY E.Name