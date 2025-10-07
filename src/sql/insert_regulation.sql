-- Insert a regulation with its embedding into the database
INSERT INTO regulations (title, content, embedding)
VALUES (%s, %s, %s::vector)
RETURNING id;
