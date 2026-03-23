CREATE TABLE IF NOT EXISTS user_profile (
                                            user_id     TEXT PRIMARY KEY,
                                            profile     TEXT NOT NULL,
                                            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);