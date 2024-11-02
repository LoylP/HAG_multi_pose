SKELETON_EDGES = {
    'face': [
        (0, 1),    # Mũi -> Mắt trái
        (0, 2),    # Mũi -> Mắt phải
        (1, 3),    # Mắt trái -> Tai trái
        (2, 4),    # Mắt phải -> Tai phải
    ],
    
    'body': [
        (0, 17),   # Mũi -> Điểm giữa vai (điểm tự tính)
        (17, 18),  # Điểm giữa vai -> Điểm giữa hông (điểm tự tính)
    ],
    
    'arms': [
        (5, 6),    # Nối 2 vai
        (5, 7),    # Vai trái -> Khuỷu tay trái
        (7, 9),    # Khuỷu tay trái -> Cổ tay trái
        (6, 8),    # Vai phải -> Khuỷu tay phải
        (8, 10),   # Khuỷu tay phải -> Cổ tay phải
    ],
    
    'legs': [
        (11, 13),  # Hông trái -> Đầu gối trái
        (13, 15),  # Đầu gối trái -> Mắt cá chân trái
        (12, 14),  # Hông phải -> Đầu gối phải
        (14, 16),  # Đầu gối phải -> Mắt cá chân phải
    ]
}