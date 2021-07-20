# 2주차
## <1>
* 장순용 강사님

<자격증>
* 국가공인 SQL개발자
* 국가공인 데이터분석 준전문가

<대회>
* 캐글
* jweek.or.kr
* DACON

##### 데이터 베이스의 구조적 특징
1. 통합성(integrated) : 동일 데이터가 중복되지 않음.  
2. 저장됨(stored) : 컴퓨터 기술을 바탕으로 저장 매체에 저장됨  
3. 공용성(shared) : 여러 사용자가 다양한 목적으로 사용할 수 있음  
4. 변화성(changeable) : 새로운 데이터의 삽입, 기존 데이터의 삭제 및 변경 가능

#### 데이터 베이스의 종류
1. 계층형 DB : 트리 구조를 이용해서 상호 관계를 계층적으로 정의함.  
2. 네트워크형 DB : 그래프 구조를 이용해서 상호 관계를 계층적으로 정의함.  
3. 관계형 DB : 계층형과 네트워크형의 복잡한 구조를 단순화 시킨 모델. 단순한 표(table)를 이용하여 데이터의 상호관계를 정의함.  
4. 객체 지향형 DB : 객체의 개념 도입. 멀티미디어와 같이 복잡한 관계를 가진 데이터를 표현하는데 효과적임.

#### SQL(Structured Query Language)
1. 데이터 정의(DDL) : 데이터 베이스를 정의, 생성, 변경, 삭제하는 것.  
2. 데이터 조작(DML) : 질의(query)를 통해서 데이터를 불러오고 처리 하는 것.  
3. 데이터 제어(DCL) : 데이터의 보안, 무결성, 병행 수행 제어 등을 하는 것

* 경량 데이터 베이스 : SQLite(1인 사용자 전제)  
* 데이터 베이스 서버 구축용 : MySQL(다수의 사용자 전제)

#### 데이터 베이스의 활용 : CRM, SCM, ERP, BI, KMS  
* 데이터베이스를 여러 테이블로 분산해 놓으면 효율적이다.

```sqlite
-- -- 우리는 SQLite 사용
-- SQL과 데이터베이스
-- SQL 기초 문법 : 행 가져오기
-- #SQL은 대소문자를 구분하지 않는다.
SELECT'Hello, World';   --문자열 출력
SELECT 1+2; --간단한 연산
SELECT *FROM Country; --'Country'라는 이름의 테이블 출력
SELECT *FROM Country ORDER BY Name; --Name으로 정렬
SELECT Name,LifeExpectancy AS 'LIFE' FROM Country; --Name출력과 LIfe는이름 바꿔서 출력

SELECT COUNT(*) FROM Country;   --행수 카운팅
--count()함수로 *는 모든 값을 의미, 변수를 넣어도 된다.
SELECT *FROM Country ORDER BY Name LIMIT 5; --출력수 제한
SELECT *FROM Country ORDER BY Name LIMIT 5 OFFSET 5; --처음 5개를 건너뛰고 제한된 5개의 값 출력
```
<hr>

## <2>
```sqlite
--특정 컬럼 가져오기
SELECT*FROM Country ORDER BY Code; 
SELECT Name, Code, Region, Population FROM Country ORDER BY Code;  
SELECT Name, Code, Region, Population FROM Country ORDER BY Population DESC;  --DESC는 내림차순
SELECT Name AS Country,Code FROM Country ORDER BY Code;   --컬럼을 불러와서 이름을 변경하여 출력할 수 있다.
SELECT Name AS Coutry,Population/1000 AS "Pop(1000s)"FROM Coutry; --간단한 수식을 적용할 수도 있다.

--SELECT + ORDER BY를 사용해서 정렬된 결과를 보여줄 수 있다.
SELECT Name FROM Country; --정렬x
SELECT Name FROM Country ORDER BY Name; --오름차순 정렬 default
SELECT Name FROM Country ORDER BY Name DESC; --내림차순 정렬
SELECT Name FROM Country ORDER BY NAme ASC;  --오름차순 정렬
SELECT Name,Continent FROM Country ORDER BY Continent,Naem; --두 개 이상의 컬럼으로 정렬
SELECT Name,Continent FROM Country ORDER BY Continent DESC, Name; --먼저 나오는 컬럼을 우선해서 정렬

--SELECT + WHERE를 사용해서 필터링 결과를 보여줄 수 있다.
--AND, OR 등으로 조건문을 연결할 수 있다.
--BETWEEN n1 AND n2 로 n1~n2로 레인지에 속하는 값만 필터링 할 수 있다.
SELECT Name, Continent, Population FROM Country WHERE Population <100000 ORDER BY Population DESC;
SELECT Name, Continent, Population FROM Country WHERE Population < 100000 AND Continent = 'Oceania' ORDER BY Population DESC;
SELECT Name, Population FROM Country WHERE Population BETWEEN 1000000 AND 10000000 ORDER BY Population DESC;
SELECT Name, Population FROM Country WHERE Population NOT BETWEEN 1000000 AND 10000000 ORDER BY Population DESC;

--SELECT + WHERE + LIKE을 사용해서 필터링 결과를 보여줄 수 있다.
--'%'는 불특정 문자열 부분을 의미하고 '_'는 불특정 문자를 의미한다.
SELECT Name,Continent FROM Country WHERE Name LIKE '%island';
SELECT Name,Continent FROM Country WHERE Name LIKE 'island%';
SELECT Name,Continent FROM Country WHERE Name LIKE '%island%';
SELECT Name,Continent FROM Country WHERE Name LIKE '_a%';   --두번째 문자가 'a'

--SELECT + WHERE + IN을 사용해서 멤버쉽 필터링 결과를 보여줄 수 있다.
SELECT Name,Continent FROM Country WHERE Continent IN ('Europe','Asia');
SELECT Name,Continent,Region FROM Country WHERE Region IN ('Western Europe','Eastern Europe') ORDER BY Region;
SELECT Name,Continent,Region FROM Country WHERE Region='Western Europe' OR Region='Eastern Europe' ORDER BY Region; 

--DISTINCT 키워드를 사용해서 중복을 제거하고 출력할 수 있다.
SELECT CountryCode, Name FROM City;
SELECT DISTINCT CountryCode FROM City;
SELECT COUNT(CountryCode)FROM City;
SELECT COUNT(DISTINCT CountryCode) FROM City;

--SELECT로 출력된 결과를 "가상"의 테이블로 사용하여 SELECT 문을 만들 수 있다. 
SELECT Name FROM (SELECT Code, Name, Continent FROM Country);
SELECT COUNT(*) FROM (SELECT DISTINCT CountryCode FROM City);

```

```sqlite
--CREATE TABLE로 데이터 베이스에 테이블을 추가할 수 있다. 
CREATE TABLE test (a INT, b TEXT, c TEXT);   --'a','b','c'컬럼의 테이블 생성
CREATE TABLE IF NOT EXISTS test (a INT,b TEXT, c TEXT);     --'a','b','c'컬럼의 테이블 중복 체크 후 생성
SELECT name FROM sqlite_master WHERE TYPE ='table'  --테이블 목록(SQLite)
PRAGMA table_info(test);  --특정 테이블에 대한 정보

--INSERT INTO ~ VALUES로 행을 추가할 수 있다.
INSERT INTO test VALUES (1, 'This','Right here!'); --1행 추가
INSERT INTO test (b,c) VALUES ('That','Over there!'); --'a'는 NULL , b와 c에 values추가
SELECT *FROM test;
--다른 테이블에서 SELECT로 가져온 결과로 행 추가 가능하다.
INSERT INTO test(a,b,c) SELECT id,name,description FROM item; --'item'테이블에서 가져온 결과를 test에 추가

--UPDATE + SET로 특정 행의 특정 컬럼 값을 변경할 수 있다.
UPDATE test SET c='Extra funny.' WHERE a=2;    --'c'컬럼의 값 변경
SELECT*FROM test;
UPDATE test SET c=NULL WHERE a=2;  --NULL값으로 변경
UPDATE test SET(b,c)=('T-Rex','Toy dinosaur')WHERE a=2;   --값 변경

--DELETE FROM으로 특정 행을 삭제할 수 있따.
DELETE FROM test WHERE a=2;     --조건에 맞는 행 삭제
DELETE FROM test;   --모든 행 삭제 , 테이블은 남음

--DROP TABLE로 특정 테이블을 삭제할 수 있다.
DROP TABLE test; --데이터베이스에서 'test'테이블 삭제
DROP TABLE IF EXISTS test; --존재여부 확인하고 데이터 베이스에서 'test'테이블 삭제

--NULL은 값이 아니라 결측치를 의미하여 0 또는 ''(공백)과는 다르다.
CREATE TABLE test (a INT, b TEXT, c TEXT);
INSERT INTO test(b,c) VALUES ('This','That');  --'a'에는 NULL을 넣어줌
SELECT *FROM test WHERE a=NULL; --작동 안함
SELECT*FROM test WHERE a IS NULL; --작동함
SELECT*FROM test WHERE a IS NOT NULL; --작동함
UPDATE test SET a=0 WHERE a IS NULL; --'a'컬럼의 NULL을 0값으로 채워 넣음.

```
<hr>

## <3>
* CRUD - Create, Read, Update, Delete
```sqlite
--ROUND함수(,)
SELECT Name AS Country,Round(Population/SurfaceArea,1) As "Population Density" FROM Country ORDER BY Population;

--서로 다르게 정렬
SELECT Name,Continent FROM Country ORDER BY Continent DESC,Name;

--단계별로 select
SELECT Name,Code FROM(SELECT Code,Name, Continent FROM COUntry);

--PRIMARY KEY AUTOINCREMENT
PRIMARY KEY AUTOINCREMENT   #PRIMARY KEY = unique not null  #autoincrement  --최고 높은거 다음으로 1씩 증가
```
* SQLite의 "기초 자료형" 5가지
1. NULL : 결측치  
2. INTEGER : 8바이트로 표현할 수 있는 정수  
3. REAL : 8바이트로 표현할 수 있는 실수  
4. TEXT : 길이에 제약이 없는 문자열  
5. BLOB : 입력된 자료를 원형 그대로 저장  
* 부울형 storage class는 없다.  
* CHAR(n)은 n개의 문자 길이로 고정된 문자열 의미  
* VARCHAR(n)은 최대 n개의 문자까지 유동적인 길이의 문자열 의미  
```sqlite
--데이터 베이스 & 테이블의 생성과 변경
CREATE TABLE myTB(a INT, b TEXT);     --새롭게 테이블 생성(및 추가)
SELECT*FROM sqlite_master WHERE TYPE="table"; --SQLite에서 테이블 목록 확인
INSERT INTO myTB VALUES(1,'foo');
SELECT*FROM myTB;
DROP TABLE myTB;  --테이블 삭제
SELECT*FROM customer; --'customer'테이블 출력
CREATE TABLE customer2 AS SELECT id,name,zip from customer; --'customer'바탕으로 'customer2'생성

#테이블 생성시 컬럼의 자료형을 명시할 수 있다. CHAR,VARCHAR = TEXT형
-- CHAR(n) = 길이 고정적, VARCHAR(n) = 최대 길이가 n까지
CREATE TABLE test(
    id INTEGER,
    name VARCHAR(255),
    address VARCHAR(255),
    city VARCHAR(255),
    state CHAR(2),
    zip CHAR(10)
                    );
INSERT INTO test VALUES(1,"John","Main ST.12","Seattle","WA",92105);
INSERT INTO test VALUES('A',"John","Main ST.123","Seattle","WA",92100); --중복도 가능하고, SQLite에서는 다른 자료형을 넣어도 작동한다.


CREATE TABLE test(
    a INTEGER NOT NULL,    --'a'에 NULL을 허용하지 않는다.
    b VARCHAR(255)
                  );
INSERT INTO test VALUES(1,"John");
INSERT INTO test(a,b) VALUES (NULL,"DAVID"); --NULL은 안된다. 오류
INSERT INTO test(b) VALUES ('PAUL');   --오류


CREATE TABLE test(
    a INTEGER NOT NULL DEFAULT 0,           --'a'에는 default값 설정
    b VARCHAR(255) UNIQUE NOT NULL          --'b'에는 중복 허용x
                );
INSERT INTO test (a,b) VALUES (3,"SALLY");
INSERT INTO test (b) VALUES('PAUL');        --'a'에 default값 0이 들어간다.
INSERT INTO test (a,b) VALUES (NULL,'David');  --오류
INSERT INTO test VALUES (4,"SALLY");          --오류, 중복 허용x


CREATE TABLE test(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a VARCHAR(255),
    b VARCHAR(255)  
                );
INSERT INTO test(a,b)VALUES('one','two');    --id로 1 자동 삽입
INSERT INTO test VALUES (5,'five','six');
INSERT INTO test(a,b)VALUES('seven','eight');   --id로 6자동 삽입  
```
<hr>

# 3주차
## <1>
```sqlite
--SQLite는 부울이 없는 대신 integer로 대체

CREATE TABLE test (
		id INTEGER PRIMARY KEY AUTOINCREMENT,     
		a VARCHAR(255),
    		b VARCHAR(255)
		);
SELECT name FROM sqlite_master WHERE type = 'table';
SELECT*FROM sqlite_sequence;
--sqlite_sequence에 autoincrement가 적용된 열의 최대값이 저장
--나중에 autoincrement가 적용될때 sqlite_sequence+1이 부여된다.
```
```sqlite
--테이블을 생성한 후 구조를 바꾼다는 것 = 열을 추가하는 거(행 추가,삭제는 구조를 바꾼다고 하지 않는다)
CREATE TABLE test (
		id INTEGER PRIMARY KEY AUTOINCREMENT,       
		a VARCHAR(255),
    		b VARCHAR(255)
		);
ALTER TABLE test ADD COLUMN c VARCHAR(100);      --테이블이 생성된 후 열을 추가하는 법 
INSERT INTO test VALUES ( 1, "John", "Male", "True");
INSERT INTO test VALUES ( 2, "Mary", "Female", "False");

--sql lite에서는 열(컬럼)을 삭제할 수 없다. but, test를 바탕으로 새로운 테이블을 만들 수 있다.
CREATE TABLE test2 AS SELECT id, a, b FROM test;
DROP TABLE test;
ALTER TABLE test2 RENAME TO test;  --새 테이블 이름 변경
SELECT * FROM test;
```
```sqlite
--BOOLEAN 자료형 표현
--0,1까지만 자료형을 받는다.
CREATE TABLE myTB (
		aColumn BOOLEAN NOT NULL CHECK (aColumn IN (0,1))     #0,1이외의 값은 reject되도록 정의
		);
INSERT INTO myTB VALUES (0);	
INSERT INTO myTB VALUES (1);		
INSERT INTO myTB VALUES (2);	  --ERROR
```
```sqlite
--FOREIGN KEY를 사용하여 테이블 사이의 관계 설정.
--일종의 제약 (constraint) 역할을 함.
CREATE TABLE Franchisee (
  FranchiseeID    INTEGER PRIMARY KEY, 
  FranchiseeName  TEXT NOT NULL
);
INSERT INTO Franchisee VALUES (111,"John");		--데이터를 입력해 둔다.
INSERT INTO Franchisee VALUES (112,"Paul");		
INSERT INTO Franchisee VALUES (113,"David");

--Store 테이블의 FranchiseeID와 Franchisee 테이블의 FranchiseeID 연결.
CREATE TABLE Store (
  StoreID     INTEGER PRIMARY KEY, 
  StoreName   TEXT NOT NULL,
  FranchiseeID INTEGER NOT NULL,
  FOREIGN KEY(FranchiseeID) REFERENCES Franchisee(FranchiseeID)
);   
INSERT INTO Store VALUES (1,"711 Gangnam", 111);
INSERT INTO Store VALUES (2,"711 Munjeong", 111);
INSERT INTO Store VALUES (3,"711 Suyu", 113);	
INSERT INTO Store VALUES (4,"711 Youido", 114);    --ERROR - 114는 Franchisee id에 없다.
DROP TABLE Franchisee;     --Franchisee먼저 삭제할 수 없다.(데이터를 넘겨주고 있기 때문)
DROP TABLE Store;          --먼저 삭제 가능

-- + id는 index로 unique하도록 사용
```
```sqlite
--테이블 JOIN
--왼쪽 테이블과 오른쪽 테이블의 공통 값(ON조건에 해당하는 값)으로 INNER JOIN
SELECT a.name, b.item_id, b.quantity, b.price FROM customer AS a INNER JOIN sale AS b ON a.id = b.customer_id;

--LEFT JOIN   왼쪽행은 모두 가져오고 오른쪽 행은 ON조건에 해당하는 것만 가져온다.
SELECT a.name, b.item_id, b.quantity, b.price FROM customer AS a LEFT  JOIN sale AS b ON a.id = b.customer_id;

--SQLite에서는 RIGHT JOIN과 FULL OUTER JOIN 지원하지 않는다. 
-- -> 데이터의 위치를 바꿔줌으로써 left join만으로 right join 효과..

--#세가지 열을 합칠 때
--1. innerjoin을 이어서
SELECT a.name,b.item_id,b.quantity,b.price,c.name as 'item_name',c.description FROM customer AS a INNER JOIN sale AS b
INNER JOIN item AS c ON a.id=b.customer_id AND c.id=b.item_id;

--2. query 안에 query 
SELECT와 INNER JOIN (SELECT)
--3. view를 이용
CREATE ViEW myView AS SELECT a.name,b.item_id,b.quantity,b.price FROM customer AS a INNER JOIN sale AS b ON a.id=b.customer_id;
SELECT*FROM myView As a INNER JOIN item AS b ON a.item_id=b.id;
```
```sqlite
--view는 select query자체를 저장
CREATE VIEW myView1 AS SELECT Name AS Country, Population / 1000 AS "Pop (1000s)" FROM Country;
CREATE VIEW myView2 AS SELECT Name, Continent, Population FROM Country WHERE Population < 100000 ORDER BY Population DESC;
CREATE VIEW myView3 AS SELECT Name, Continent FROM Country WHERE Name LIKE '_a%';

SELECT * FROM myView1;
SELECT * FROM myView2;
SELECT Name as CountryName FROM myView3;

DROP VIEW myView1;
DROP VIEW myView2;
DROP VIEW myView3;
```
<hr>

## <2>
### SQL 함수  
SQL은 1부터 센다 (ex : python은 0부터
```sqlite
--문자열 함수
--x || y : 문자열 x와 y연결
SELECT Code || ' = ' || Name FROM Country;

--LENGTH(x) : 문자열 x의 길이
SELECT Name, LENGTH(Name) FROM Country;

--SUBSTR(x,n,m) : x=문자열, n=시작 위치, m =반환 문자 개수
SELECT Name, SUBSTR(Name,1,3) FROM Country;     --첫번째 위치에서 세자리 반환

--TRIM(x) : 문자열 x에서 스페이스 제거
--TRIM(x,y) : 문자열 x에서 문자 y 제거
--LTRIM(),RTRIM() : TRIM()과 유사, 단 left 또는 right에 국한되게 문자 제거
SELECT TRIM("   <===>    ");   

--UPPER(x),LOWER(x) : 문자열 x의 대문자화 또는 소문자화
SELECT Name, UPPER(Name) FROM Country; 
SELECT Name, LOWER(Name) FROM Country; 

--REPLACE(x,y,z) : 문자열 x에서 패턴 y를 패턴 z로 대체
```
```sqlite
--집계(aggregation)하지 않는 계산용 함수와 연산자
SELECT ROUND(Population/SurfaceArea,0) AS PopDensity FROM Country;    --나누기 & 반올림.
SELECT 7 % 4;				  	      --x%y : 나머지.
SELECT ABS(-123);				  	 --ABS(x) : 절대값.
SELECT ROUND(123.456789, 3);           --ROUND(x,n) : x소수점 이하 n자리까지 , 반올림.

```
```sqlite
--집계 함수와 순위 함수를 통 털어서 윈도우 함수 (Window Function)라 부른다.
--집계 (aggregation) 함수.
SELECT COUNT(*) FROM Country;                --행수 카운트.
SELECT COUNT(Population) FROM Country;     --NULL이 아닌 Population 값의 개수.
SELECT Continent, COUNT(*) FROM Country GROUP BY Continent;  	-- 대륙별 국가의 수.
SELECT Region, COUNT(*) FROM Country GROUP BY Region;  	--Region별 국가의 수.
SELECT COUNT(DISTINCT Continent) FROM Country;     		-- 고유한 대륙의 개수. 

--SUM(X) : 그룹별 X컬럼의 합을 구해준다.
--AVG(X) : 그룹별 X컬럼의 평균을 구해준다.
SELECT AVG(Population) FROM Country WHERE Region = 'Western Europe';  -- 서유럽 지역의 국가의 인구 평균.

--MIN(x),MAX(X) : 그룹별 X컬럼의 최소값, 최대값 계산
SELECT MIN(Population), MAX(Population) FROM Country WHERE Region = 'Western Europe'; 

```
```sqlite
-- 날짜와 시간 관련 함수. 
SELECT DATE('now');		--현재 날짜.
SELECT TIME('now');		--현재 시각. 
SELECT DATETIME('now');		--날짜와 시각.
SELECT STRFTIME('%Y', 'now');		-- 연도
SELECT STRFTIME('%m', 'now');	  -- 월(1~12)
SELECT STRFTIME('%d', 'now');		-- 일
SELECT STRFTIME('%w', 'now');   -- 요일(0~6,0=일요일)	 
SELECT STRFTIME('%s', 'now');   -- 초
SELECT TIME('now','+9 hour','-30 minute');   --now나 datetime에 +와 -를 할 수도 있다.
```
### python에서 sqlite 접근
```python
import sqlite3
import pandas as pd
import os
from google.colab import drive
# 구글 드라이브 마운트.
drive.mount('/content/drive')  # 절차를 따라서 한다
# 현 폴더의 위치를 보여준다.
!pwd        # Linux
from platform import platform
platform()  #platform확인 : Linux-5.4.104+-x86_64-with-Ubuntu-18.04-bionic
# 작업 폴더로 이동한다.
# 데이터 파일은 이미 올려져 있어야 한다.
os.chdir("/content/drive/MyDrive/광인사/01_SQL과 데이터 베이스_배포/")
# %cd /content/drive/MyDrive/광인사/01_SQL과 데이터 베이스_배포/        #디렉토리 경로이동 
# 현 폴더의 위치를 보여준다.
!pwd
# 현 폴더의 내용을 보여준다.
!ls
```
```python
# SQLite 에 연결 & Cursor 생성.
os.chdir("../database/")
conn = sqlite3.connect("world.db")
cur = conn.cursor()
# MySQL 행 가져오기.
sql = "SELECT Name, Code, Region, Population FROM Country ORDER BY Population DESC;" #문자열
cur.execute(sql);       #문자열을 실어보내서 실행
res = cur.fetchall()      #data를 tuple형태로 list에 저장
                          #한번 fetchall하면 fetch하고 내용이 클리어된다.
# 자료형은 List of tuples.
print(type(res))  #<class 'list'>
print(type(res[0]))  #<class 'tuple'>
```
```python
for x in res:
    print(x)

# ('China', 'CHN', 'Eastern Asia', 1277558000)
# ('India', 'IND', 'Southern and Central Asia', 1013662000)
# ('United States', 'USA', 'North America', 278357000)
# ('Indonesia', 'IDN', 'Southeast Asia', 212107000)
# ('Brazil', 'BRA', 'South America', 170115000)
# ('Pakistan', 'PAK', 'Southern and Central Asia', 156483000)
# ('Russian Federation', 'RUS', 'Eastern Europe', 146934000)
# ('Bangladesh', 'BGD', 'Southern and Central Asia', 129155000)
# ('Japan', 'JPN', 'Eastern Asia', 126714000)
# ('Nigeria', 'NGA', 'Western Africa', 111506000)
# ('Mexico', 'MEX', 'Central America', 98881000)
# ('Germany', 'DEU', 'Western Europe', 82164700)

conn.close()
```
```python
# SQLite 에 연결 & Cursor 생성.
os.chdir("../database")       #현재 디렉토리를 새로운 위치의 디렉토리로 변경한다.
conn = sqlite3.connect("scratch.db")
cur = conn.cursor()         #커서는 내장 SQL문의 수행결과로 반환될 수 있는 복수의 튜플들을 액세스 할 수 있도록 해주는 개념

# MySQL 테이블 생성.
sql = "CREATE TABLE IF NOT EXISTS test ( a INT, b TEXT, c TEXT );"
cur.execute(sql);     #SQL문 실행
conn.commit();      #모든 작업을 정상적으로 처리하겠다고 확정하는 명령어, 데이터베이스 자체에 저장,변경

# "test" 테이블에 행 추가.
sql = "INSERT INTO test VALUES ( ?, ?, ? );"      # ?를 placeholder로 사용!
x = [1, "Hello", "World"]                        # 삽입될 내용.
cur.execute(sql,x);                                # 실행!
conn.commit();

# "test" 테이블에 행 계속 추가.
sqls = ["INSERT INTO test VALUES ( 2, 'This', 'Right here!' );", 
        "INSERT INTO test ( b, c ) VALUES ( 'That', 'Over there!' );",
        "INSERT INTO test ( a, b, c ) SELECT id, name, description FROM item;"] 
for a_sql in sqls:
    cur.execute(a_sql);
conn.commit();

# 테이블 내용 보기.
sql = "SELECT * FROM test"
cur.execute(sql);
res = cur.fetchall()    #fetch all -> 레코드를 배열형식으로 저장해준다.
for x in res:
    print(x)

sql = "DROP TABLE test;"
cur.execute(sql);
conn.commit()
conn.close()
```
```python
##2
#python에는 SQLite가 이미 설치되어 있다.(Colab or Anaconda)
#커서(cursor)객체를 만들고 SQL 커맨드를 쉽게 처리할 수 있다.
import sqlite3
conn=sqlite3.connect('customer.db')    #경로를 잘못 적어도 열린다(새로 생성된다) 주의!
cur=conn.cursor()                     #커서 객체 cursor 메서드는 단일 선택적 매개 변수 factory를 받아들인다.
cur.execute("""CREATE TABLE customers(
  first_name  text, last_name text, email text
)""")                                   #Multi-line
conn.commit()  #변화가 있는 경우만 commit
conn.close()   #닫아야만 파일이 저장됨, 데이터 베이스 연결 종료
```
```python
import sqlite3
conn = sqlite3.connect('customer.db')
curr=conn.cursor()
sql = "INSERT INTO test VALUES (?,?,?);"      #?를 placeholder로 사용
x = [2,"This","That"]                         #삽입될 행(list 또는 tuple)
cur.execute(sql,x)                            #실행
conn.commit()
conn.close()
```
<hr>

## <3>
### SQLite 데이터 베이스와 Pandas 데이터프레임
```python
import sqlite3
import pandas as pd
import os
from google.colab import drive
# 구글 드라이브 마운트.
drive.mount('/content/drive') 
!pwd        # Linux
os.chdir("/content/drive/MyDrive/광인사/01_SQL과 데이터 베이스_배포/")
!ls
```
### 1). 데이터프레임의 내용을 데이터베이스로 옮겨본다: 
#### 데이터프레임을 가져온다:
```python
os.chdir("./data")
df = pd.read_csv("data_studentlist_en.csv",header="infer")
nrows = df.shape[0]     #df.shape = (17, 8)
my_header = df.columns
df.head(3)
df.describe()
pd.set_option('precision',2)    #소수점 이하 2자리까지만 보여준다##pandas자체 설정 바꾸기
df.describe()       #다양한 통계량을 요약해준다.
# 구조를 본다.
df.info()
```
#### 테이블 내용 가져와 보기:
```python
# SQLite 데이터 베이스 생성(또는 연결) & Cursor 생성.
conn = sqlite3.connect("student.db")                         # 처음 실행시 신규 생성.
cur = conn.cursor()
# SQL 테이블 생성.
sql = "CREATE TABLE IF NOT EXISTS studentlist ( name TEXT, gender TEXT, age INT, grade INT, absence TEXT, bloodtype TEXT, height REAL, weight REAL);"
cur.execute(sql);
conn.commit();
# SQL 테이블 내용 클리어.
sql = "DELETE FROM studentlist;"
cur.execute(sql);
conn.commit();
# SQL 테이블 내용 확인.
sql = "SELECT * FROM studentlist;"
cur.execute(sql);
res = cur.fetchall();
print(res)
```
#### 한 행 씩 옮겨 담는다:
```python
# "studentlist" 테이블에 행 추가.
sql = "INSERT INTO studentlist VALUES ( ?, ?, ?, ?, ?, ?, ?, ? );"      # ?를 placeholder로 사용!
for i in range(nrows):
    x = df.loc[i].values                         # 삽입될 내용.
    x = list(x)
    x[2] = int(x[2])                               # INT64를 INT로 변환.
    x[3] = int(x[3])                               # INT64를 INT로 변환.
    cur.execute(sql,x);                            # 삽입.  
conn.commit();
# 테이블 내용 보기.
sql = "SELECT * FROM studentlist"
cur.execute(sql);
res = cur.fetchall()
for x in res:
    print(x)
```
### 2). 데이터베이스의 내용을 데이터프레임으로 옮겨본다: 
```python
# 테이블 내용 가져온다.
sql = "SELECT * FROM studentlist"
cur.execute(sql);
res = cur.fetchall()
res
# 새로운 데이터 프레임 생성 & 확인.
df_new = pd.DataFrame(data=res, columns=my_header, index=None)
df_new.head(5)
```
### 3). 뒷 정리:
```python
# 테이블 삭제.
sql = "DROP TABLE studentlist;"
cur.execute(sql);
conn.commit();
conn.close()
```

# 4주차
## <1>
#### Jupyter Notebook과 구글 Colab 사용법
#### 마크다운
#### 매직커맨드

## <2>
### 확률
#### 확률의 정의
* 시행과 사건
  - 사건, 표본공간(S)
    - 근원사건 : 하나의 값만이 나오는 사건
    - AND, OR, NOT : 교집합, 합집합, 차집합
    - 배반사건, 여사건

* 수학적 확률의 정의 : N이 한 시행에 따라서 일어날 수 있는 모든 경우의 수.
Na이 사건 A가 일어나는 경우의 수일 때 수학적 확률(P(A) = Na/N)  

* 확률의 기본성질
  - 임의의 사건 A에 대한 확률은   
  0 <= P(A) <=1 을 충족한다.
  - 근원사건의 확률 pi를 모두 더하면 1이 되어야 한다.   
  즉, 표본공간 S에 대한 확률 P(S) = 1이다.

* 통계적(경험적) 확률의 정의
  - 어떤 조건 아래에서 실험 또는 관측한 자료의 총수를 N이라 하고, 그 중에서 어떤 사건 A가 일어난 횟수를 𝑁𝐴라 할 때 상대도수는 다음과 같이 정의한다:  
상대도수 = 𝑁𝐴 / N
  - 그러면 사건 A가 일어날 통계적 확률은 다음과 같이 상대도수의 극한이다.  
  통계적 확률 = lim(𝑁→∞) NA/N

#### 확률의 덧셈
* OR  
P( 𝐴 ∪ 𝐵 )= 𝑃 (𝐴) + 𝑃(𝐵) − 𝑃(𝐴 ∩ 𝐵)
* 만약 A와 B가 동시에 일어날 수 없는 '배반사건'인 경우, 𝑃(𝐴 ∩ 𝐵) = 0 이며  
𝑃 (𝐴 ∪ 𝐵) = 𝑃 (𝐴) + 𝑃 (B)

#### 확률의 곱셈
* 조건부 확률 (Conditional Probability)
  - 확률이 0이 아닌 사건 A와 B에 대해서 사건 A가 일어났다는 전제로 사건 B가 일어날 확률을 조건부 확률이라 부르며 𝑃(𝐵|𝐴)와 같이 표기한다.   
  𝑃(𝐵|𝐴) = 𝑃(𝐴 ∩ 𝐵) / 𝑃(𝐴)  
  𝑃(𝐴 ∩ 𝐵) = 𝑃(𝐵|𝐴) 𝑃(𝐴)= 𝑃(𝐴|𝐵) 𝑃(B)

* 베이즈 정리 : 확률의 곱셈정리를 이용하여 베이즈 정리를 도출해 낼 수 있다.  
  𝑃(𝐴|𝐵) 𝑃(𝐵) = 𝑃(𝐵|𝐴) 𝑃(𝐴)
* 베이즈 정리의 활용 P(A|B) = P(B|A)P(A) / P(B|A)P(A) + P(B|A^c)P(A^c)

## <3>
####  신용 카드 사용 기록 EDA. 분석
* 자세한건 ex_credit_eda_colab.ipynb
#### 1. 데이터를 불러와서 살펴 본다.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv("../data/data_credit.csv")
df.shape
df.head(5)
```
#### 2. 데이터를 정제하고 결측치 처리를 한다.
```python
# 결측치가 분명히 있는 것 같으니 컬럼별 집계해 본다.
df.isnull().sum(axis=0)

# 일시불이 NA로 처리 되었으니 결측치를 1로 채운다.
df = df.fillna(value=1)
df.head(3)

# 결측치를 다시 집계해 본다.
df.isnull().sum(axis=0)

# Installments의 자료형을 정수로 바꾼다.
df["installments"] = df.installments.astype("int32")

# month 파생 변수를 만든다.
df["month"] = df.date.apply( lambda x : x[5:7] )

# hour 파생 변수를 만든다.
df["hour"] = df.time.apply(lambda x : x[:2])

# date와 time은 더 이상 사용하지 않겠으니 제거한다.
df.drop(columns=['date','time'], inplace=True)
df.head(3)
```
#### 3. 개개 변수별로 탐색한다.
```python
# store_id별 도수를 집계하고 히스토그램으로 출력해 본다.
df.store_id.value_counts().plot.hist(bins=50, color="turquoise",alpha=0.5)
plt.show()
# 큰 skew를 보이므로 log 변환해서 다시 시각화 해본다.
np.log(df.store_id.value_counts()).plot.hist(bins=50, color="turquoise",alpha=0.5)
plt.show()
# 중위수를 계산해 본다.
freq1 = df.store_id.value_counts() 
print(np.median(freq1))
# Top store_id를 본다.
freq1.sort_values(ascending = False)[:10]
```
#### 4. 수치형 변수 1개 + 명목형 변수 1개를 조합한 탐색.
```python
# store_id 별 결제 금액.
# 평균 금액 랭킹 top 10.
df.groupby("store_id").amount.mean().sort_values(ascending=False)[:10]
# Pivot table 방식으로 다시 해본다.
pd.pivot_table(df, index = "store_id", values="amount", aggfunc = np.mean).sort_values(by="amount", ascending=False)[:10]
# 총 금액 랭킹 top 10.
df.groupby("store_id").amount.sum().sort_values(ascending=False)[:10]
# Pivot table 방식으로 다시 해본다.
pd.pivot_table(df, index = "store_id", values="amount", aggfunc = np.sum).sort_values(by="amount", ascending=False)[:10]
```
#### 5. 수치형 변수 1개 + 명목형 변수 2개를 조합한 탐색.
```python
# 월별 시간대별 매출 추이.
# 평균.
df2 = pd.pivot_table(df, index="hour", columns="month", values = "amount", aggfunc=np.mean)
df2
# 시각화.
df2.plot()
plt.show()
```
#### 6. 등급화.
```python
# store_id별 total 결제 금액에 의한 3 등급 분류. HINT: pd.qcut() 사용.
ser = df.groupby("store_id").amount.sum()
ser2 = pd.qcut(ser,3,labels=['Low', 'Medium', 'High'])                      # qcut() 함수 사용!
combo = pd.DataFrame({"Store ID":ser.index, "Amount":ser.values, "Category":ser2.values})
combo.head(10)
# 등급별 평균.
combo.groupby("Category")["Amount"].mean()
```


<hr>