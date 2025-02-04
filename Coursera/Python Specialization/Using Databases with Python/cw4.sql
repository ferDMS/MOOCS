DROP TABLE IF EXISTS cw4_Member;
DROP TABLE IF EXISTS cw4_User;
DROP TABLE IF EXISTS cw4_Course;

CREATE TABLE cw4_Course (
    id INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
    title VARCHAR(50)
);

CREATE TABLE cw4_User (
    id INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE cw4_Member (
    user_id INT FOREIGN KEY REFERENCES cw4_User(id),
    course_id INT FOREIGN KEY REFERENCES cw4_Course(id),
    role INT NOT NULL
);