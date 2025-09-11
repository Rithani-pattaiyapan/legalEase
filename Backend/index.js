import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './db/connectDB.js';
import lawyerRouter from './router/lawyerRouter.js';
import peopleRouter from './router/peopleRouter.js';
import caseRouter from './router/caseRouter.js';
import User from './module/user.module.js';
import LawyerProfile from './module/lawyer.module.js';
import CommonUser from './module/commonUser.module.js';
import Case from './module/case.module.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
connectDB()
  .then(() => {
    console.log("Connected to MongoDB");
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err);
  });

// Routes
app.use('/lawyer', lawyerRouter);
app.use('/people', peopleRouter);
app.use('/case', caseRouter);

app.get('/users', async(req, res) => {
  try {
    const users = await User.find({});
    res.status(200).json({
      success : true,
      data : users
    });
  } catch (error) {
    res.status(500).json({
      success : false,
      message : "Server Error"
    });
  }
});

app.get('/lawyers_data', async(req, res) => {
  try {
    const profiles = await LawyerProfile.find({}).populate('user_id', 'name email');
    res.status(200).json({
      success : true,
      data : profiles
    });
  } catch (error) {
    res.status(500).json({
      success : false,
      message : "Server Error"
    });
  }
});

app.get('/common_users', async(req, res) => { 
  try {
    const commonUsers = await CommonUser.find({});
    res.status(200).json({
      success : true,
      data : commonUsers
    });
  } catch (error) {
    res.status(500).json({
      success : false,
      message : "Server Error"
    });
  }
});

app.get('/case_details', async(req, res) => {
  try {
    const cases = await Case.find({})
      .populate({
        path: 'client_id',
        populate: {
          path: 'user_id',
          select: 'name'
        }
      })
      .populate({
        path: 'lawyer_id',
        populate: {
          path: 'user_id',
          select: 'name'
        }
      });

    const result = cases.map(c => ({
      title: c.title,
      description: c.description,
      client: c.client_id?.user_id?.name, 
      lawyer: c.lawyer_id?.user_id?.name, 
      process_updates: c.process_updates
    }));

    console.log(result);


    res.json({ success: true, cases: result });

  } catch (error) {
    res.status(500).json({
      success : false,
      message : "Server Error"
    });
  }
});

// Test route

app.get('/', (req, res) => {
  res.send('API is running...');
});

// Start server
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});