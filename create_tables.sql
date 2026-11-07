CREATE TABLE users (
    id INT NOT NULL AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(320) NOT NULL,
    full_name VARCHAR(160) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'researcher',
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY ix_users_username (username),
    UNIQUE KEY ix_users_email (email),
    CONSTRAINT ck_users_role
        CHECK (role IN ('admin', 'researcher', 'clinician'))
) ENGINE=InnoDB;

CREATE TABLE revoked_tokens (
    jti VARCHAR(64) NOT NULL,
    expires_at DATETIME NOT NULL,
    revoked_at DATETIME NOT NULL,
    PRIMARY KEY (jti),
    KEY ix_revoked_tokens_expires_at (expires_at)
) ENGINE=InnoDB;

CREATE TABLE prediction_logs (
    id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    disease_id VARCHAR(40) NOT NULL,
    model_name VARCHAR(80) NOT NULL DEFAULT 'weighted_ensemble',
    positive_score FLOAT NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    KEY ix_prediction_logs_user_id (user_id),
    KEY ix_prediction_logs_disease_id (disease_id),
    CONSTRAINT fk_prediction_logs_user_id_users
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE model_registry (
    id INT NOT NULL AUTO_INCREMENT,
    disease_id VARCHAR(40) NOT NULL,
    model_name VARCHAR(80) NOT NULL,
    version VARCHAR(80) NOT NULL,
    artifact_path VARCHAR(1024) NOT NULL,
    is_active BOOLEAN NOT NULL,
    created_by_id INT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_model_registry_version (disease_id, model_name, version),
    KEY ix_model_registry_disease_id (disease_id),
    CONSTRAINT fk_model_registry_created_by_id_users
        FOREIGN KEY (created_by_id) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB;

CREATE TABLE experiment_logs (
    id INT NOT NULL AUTO_INCREMENT,
    user_id INT NULL,
    disease_id VARCHAR(40) NOT NULL,
    experiment_type VARCHAR(80) NOT NULL,
    status VARCHAR(20) NOT NULL,
    parameters JSON NULL,
    artifact_path VARCHAR(1024) NULL,
    error_message TEXT NULL,
    created_at DATETIME NOT NULL,
    completed_at DATETIME NULL,
    PRIMARY KEY (id),
    KEY ix_experiment_logs_user_id (user_id),
    KEY ix_experiment_logs_disease_id (disease_id),
    CONSTRAINT fk_experiment_logs_user_id_users
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB;

CREATE TABLE report_logs (
    id INT NOT NULL AUTO_INCREMENT,
    user_id INT NULL,
    disease_id VARCHAR(40) NOT NULL,
    report_type VARCHAR(80) NOT NULL,
    artifact_path VARCHAR(1024) NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    KEY ix_report_logs_user_id (user_id),
    KEY ix_report_logs_disease_id (disease_id),
    CONSTRAINT fk_report_logs_user_id_users
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB;