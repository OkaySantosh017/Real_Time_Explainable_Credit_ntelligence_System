const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./config/db');
const authRoutes = require('./routes/auth');
const userRoutes = require('./routes/user');
const creditRoutes = require('./routes/credit');
const errorHandler = require('./middleware/errorHandler');

require('express-async-errors');
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

connectDB();

app.get('/', (req, res) => res.json({ message: 'Credit Intelligence API is running' }));
app.use('/api/auth', authRoutes);
app.use('/api/user', userRoutes);
app.use('/api/credit', creditRoutes);

app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
