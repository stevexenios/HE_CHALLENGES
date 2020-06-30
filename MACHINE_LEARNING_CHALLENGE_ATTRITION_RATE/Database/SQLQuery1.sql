USE master
GO

ALTER DATABASE Hacker_Earth_DB_1 SET MULTI_USER

DROP DATABASE Hacker_Earth_DB_1
GO

CREATE DATABASE Hacker_Earth_DB_1
GO

USE Hacker_Earth_DB_1
GO

CREATE TABLE EMPLOYEE
(
    Employee_ID varchar(10),
    Age int,
    Education int,
    Relationship_Status varchar(10),
    Hometown varchar(10),
    [Name] varchar(30),
    Gender varchar(1)
);

CREATE TABLE SERVICE
(
    Employee_ID varchar(10),
    Unit varchar(10),
    Post_Level int,
    Time_since_promotion int,
    Time_of_service float,
    Pay_Scale float,
    Collaboration_and_Teamwork int,
    Compensation_and_Benefits varchar(10)
);


INSERT INTO EMPLOYEE (Employee_ID, Age, Education, Relationship_Status, Hometown, [Name], Gender) VALUES
('EID_7044', 47, 4,	'Married', 'Franklin', 'Paul Jones', 'M'),
('EID_11061', 43, 3, 'Married',	'Springfiel', 'Roger Reyes', 'M'),
('EID_4392', 64, 4, 'Single', 'Franklin', 'Shane Martin', 'M'),
('EID_13606', 27, 3, 'Married',	'Franklin',	'Christopher Huynh', 'M'),
('EID_7656', 45, 5, 'Married', 'Lebanon', 'Jacob Leblanc', 'M'),
('EID_8156', 44, 4, 'Single', 'Clinton', 'Colleen Strong', 'F'),
('EID_3577', 64, 4,	'Single', 'Washington',	'James Wise', 'M'),
('EID_5730', 50, 4,	'Married', 'Springfiel', 'Brenda Marquez',	'F'),
('EID_6064', 65, 1,	'Married', 'Franklin', 'Robert George',	'M');


INSERT INTO SERVICE (Employee_ID, Unit, Post_Level, Time_since_promotion, Time_of_service, Pay_Scale, Collaboration_and_Teamwork, Compensation_and_Benefits) VALUES
('EID_7044', 'Sales', 2, 1, 9, 8.0, 4, 'type4'),
('EID_11061', 'Sales',	3, 2, 14, 10.0, 4, 'type2'),
('EID_4392', 'Logistics', 3, 1, 26, 5.0, 2, 'type2'),
('EID_13606', 'Human Reso', 5, 3, 3, 6.0, 3, 'type3'),
('EID_7656', 'Purchasing',	1, 4, 7, 8.0, 3, 'type3'),
('EID_8156', 'Purchasing', 1, 1, 21, 5.0, 3, 'type2'),
('EID_3577', 'Purchasing', 5, 2, 32, 7.0, 4, 'type2'),
('EID_5730', 'Sales', 2, 4, 25, 9.0, 5, 'type2'),
('EID_6064', 'IT', 2, 2, 37, 1.0, 4, 'type2');